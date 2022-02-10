import tensorflow as tf

from tensorflow.keras.layers import (
    MultiHeadAttention,
    Dense,
    LayerNormalization
)


class RFF(tf.keras.layers.Layer):
    def __init__(
        self,
        units_per_layers: List[int],
        dense_kwargs: Dict = dict(activation="relu"),
        **kwargs
    ):
        """Row-wise Feed-Forward (rFF) layer.
        Processes each row independently using an mlp
        y = MLP(x)

        Parameters
        ----------
        units_per_layers : List[int]
            units per Dense layer
        dense_kwargs : Dict, optional
            aditional kwargs for Dense layers,
            by default {"activation": "relu"}
        """
        super().__init__(**kwargs)

        self.mlp = tf.keras.Sequential(
            layers=[
                Dense(
                    units=units,
                    **dense_kwargs
                )
                for units in units_per_layers
            ]
        )

    def call(
        self,
        x: tf.Tensor
    ) -> tf.Tensor:
        """y = MLP(x)

        Parameters
        ----------
        x : tf.Tensor
            shape: (batch_size, n, d)

        Returns
        -------
        tf.Tensor
            shape: (batch_size, n, d)
        """
        return self.mlp(x)


class MAB(tf.keras.layers.Layer):
    def __init__(
        self,
        attention_axes: Tuple[int, ...],
        multi_head_attention_kwargs: Dict,
        rff_kwargs: Dict,
        layer_normalization_h_kwargs: Dict,
        layer_normalization_out_kwargs: Dict,
        normalize: bool = True,
        **kwargs
    ):
        """Multihead Attention Block (MAB)
        MAB(X, Y) = LayerNorm(H + rFF(H))
        where H = LayerNorm(X + Multihead(X, Y, Y))

        Parameters
        ----------
        attention_axes : Tuple[int, ...]
            determines the axes on which attention is performed
        multi_head_attention_kwargs : Dict
            keras layer kwargs
        rff_kwargs : Dict
            RFF class kwargs
        layer_normalization_h_kwargs : Dict
            keras layers kwargs
        layer_normalization_out_kwargs : Dict
            keras layer kwargs
        normalize : bool, optional
            if True, use layer_normalization layers as part of
            the architecture (as per the original paper),
            by default True
        """
        super().__init__(**kwargs)

        self.multi_head = MultiHeadAttention(
            **multi_head_attention_kwargs,
            attention_axes=attention_axes
        )

        self.rff = RFF(
            **rff_kwargs
        )

        self.normalize = normalize
        if normalize:
            self.layer_norm_h = LayerNormalization(
                **layer_normalization_h_kwargs
            )

            self.layer_norm_out = LayerNormalization(
                **layer_normalization_out_kwargs
            )

    def call(
        self,
        x: tf.Tensor,
        y: tf.Tensor
    ) -> tf.Tensor:
        """MAB(X, Y) = LayerNorm(H + rFF(H))
        where H = LayerNorm(X + Multihead(X, Y, Y))

        Parameters
        ----------
        x : tf.Tensor
            Query, shape = (batch_size, n, d)
        y : tf.Tensor
            Key and Value, shape = (batch_size, n', d)

        Returns
        -------
        tf.Tensor
            shape = (batch_size, n, d)
        """
        h = x + self.multi_head(x, y, y)
        if self.normalize:
            h = self.layer_norm_h(h)

        out = h + self.rff(h)
        if self.normalize:
            out = self.layer_norm_out(out)

        return out


class SAB(MAB):

    def call(
        self,
        x: tf.Tensor
    ) -> tf.Tensor:
        """SAB(X) = MAB(X, X)

        Parameters
        ----------
        x : tf.Tensor
            Query, Key, Value, shape = (batch_size, n, d)

        Returns
        -------
        tf.Tensor
            shape = (batch_size, n, d)
        """
        return super().call(x, x)


class ISAB(tf.keras.layers.Layer):

    def __init__(
        self,
        attention_axes: Tuple[int, ...],
        m: int,
        d: int,
        mab_h_kwargs: Dict,
        mab_out_kwargs: Dict,
        **kwargs
    ):
        """Induced Self Attention Block
        ISAB (X) = MAB(X, H) ∈ R n×d
        where H = MAB(I, X) ∈ R m×d

        Parameters
        ----------
        attention_axes : Tuple[int, ...]
            determines the axes on which attention is performed
        m : int
            number of inducing points
        d : int
            data point dimension
        mab_h_kwargs : Dict
            MAB block kwargs
        mab_out_kwargs : Dict
            MAB block kwargs
        """
        super().__init__(**kwargs)

        self.i = self.add_weight(
            name="inducing_points",
            shape=(m, d),
            initializer="random_normal",
            trainable=True
        )

        self.mab_h = MAB(
            **mab_h_kwargs,
            attention_axes=attention_axes
        )

        self.mab_out = MAB(
            **mab_out_kwargs,
            attention_axes=attention_axes
        )

    def call(
        self,
        x: tf.Tensor
    ) -> tf.Tensor:
        """ISAB (X) = MAB(X, H) ∈ R n×d
        where H = MAB(I, X) ∈ R m×d

        Parameters
        ----------
        x : tf.Tensor
            shape (batch_size, n, d)

        Returns
        -------
        tf.Tensor
            shape (batch_size, n, d)
        """
        batch_shape = x.shape[:-2]

        repeated_i = self.i
        for size in reversed(batch_shape):
            repeated_i = tf.repeat(
                tf.expand_dims(
                    repeated_i,
                    axis=0
                ),
                (size if size is not None else 1,),
                axis=0
            )
        h = self.mab_h(repeated_i, x)
        return self.mab_out(x, h)


class PMA(tf.keras.layers.Layer):

    def __init__(
        self,
        attention_axes: Tuple[int, ...],
        k: int,
        d: int,
        rff_kwargs: Dict,
        mab_kwargs: Dict,
        **kwargs
    ):
        """Pooling by Multihead Attention block
        PMA(Z) = MAB(S, rFF(Z)) ∈ R k×d
        where S ∈ R k×d

        Parameters
        ----------
        attention_axes : Tuple[int, ...]
            determines the axes on which attention is performed
        k : int
            number of seed vectors
        d : int
            data point dimension
        rff_kwargs : Dict
            RFF kwargs
        mab_kwargs : Dict
            MAB kwargs
        """
        super().__init__(**kwargs)

        self.k = k
        self.s = self.add_weight(
            name="seed_vectors",
            shape=(k, d),
            initializer="random_normal",
            trainable=True
        )

        self.rff = RFF(
            **rff_kwargs
        )

        self.mab = MAB(
            **mab_kwargs,
            attention_axes=attention_axes
        )

    def call(
        self,
        z: tf.Tensor
    ) -> tf.Tensor:
        """PMA(Z) = MAB(S, rFF(Z)) ∈ R k×d
        where S ∈ R k×d

        Parameters
        ----------
        z : tf.Tensor
            shape (batch_size, n, d)

        Returns
        -------
        tf.Tensor
            shape (batch_size, k, d)
        """
        batch_shape = z.shape[:-2]

        repeated_s = self.s
        for size in reversed(batch_shape):
            repeated_s = tf.repeat(
                tf.expand_dims(
                    repeated_s,
                    axis=0
                ),
                (size if size is not None else 1,),
                axis=0
            )
        return self.mab(repeated_s, self.rff(z))


class Encoder(tf.keras.layers.Layer):

    def __init__(
        self,
        type: str,
        attention_axes: Tuple[int, ...],
        kwargs_per_layer: List[Dict],
        **kwargs
    ):
        """Set Transformer encoder
        Stack of SAB or ISAB blocks
           Encoder(X) = SAB(SAB(... X))
        or Encoder(X) = ISAB(ISAB(... X))

        Parameters
        ----------
        type : str
            one of ["SAB", "ISAB"]
        attention_axes : Tuple[int, ...]
            determines the axes on which attention is performed
        kwargs_per_layer : List[Dict]
            kwargs for SAB or ISAB class

        Raises
        ------
        AssertionError
            if type is not in ["SAB", "ISAB"]
        """
        super().__init__(**kwargs)

        if type not in ["SAB", "ISAB"]:
            raise AssertionError(
                "type should be one of [`SAB`, `ISAB`]"
            )
        self.seq = tf.keras.Sequential(
            layers=[
                (
                    SAB
                    if type == "SAB"
                    else ISAB
                )
                (
                    **layer_kwargs,
                    attention_axes=attention_axes
                )
                for layer_kwargs in kwargs_per_layer
            ]
        )

    def call(
        self,
        x: tf.Tensor
    ) -> tf.Tensor:
        """Encoder(X) = SAB(SAB(... X))
        or Encoder(X) = ISAB(ISAB(... X))

        Parameters
        ----------
        x : tf.Tensor
            shape (batch_size, n, d)

        Returns
        -------
        tf.Tensor
            shape (batch_size, n or m, d)
        """
        return self.seq(x)


class Decoder(tf.keras.layers.Layer):

    def __init__(
        self,
        attention_axes: Tuple[int, ...],
        pma_kwargs: Dict,
        sab_kwargs: Dict,
        rff_kwargs: Dict,
        **kwargs
    ):
        """Set Transformer Decoder
        Decoder(Z) = rFF(SAB(PMA(Z))) ∈ R k×d

        Parameters
        ----------
        attention_axes : Tuple[int, ...]
            determines the axes on which attention is performed
        pma_kwargs : Dict
            PMA kwargs
        sab_kwargs : Dict
            SAB kwargs
        rff_kwargs : Dict
            RFF kwargs
        """
        super().__init__(**kwargs)

        self.attention_axes = attention_axes
        self.pma = PMA(
            attention_axes=attention_axes,
            **pma_kwargs
        )

        self.sab = SAB(
            attention_axes=attention_axes,
            **sab_kwargs
        )

        self.rff = RFF(
            **rff_kwargs
        )

    def call(
        self,
        z: tf.Tensor
    ) -> tf.Tensor:
        """Decoder(Z) = rFF(SAB(PMA(Z))) ∈ R k×d

        Parameters
        ----------
        z : tf.Tensor
            shape (batch_size, n, d)

        Returns
        -------
        tf.Tensor
            shape (batch_size, k, d)
        """
        return self.rff(
            self.sab(
                self.pma(z)
            )
        )
