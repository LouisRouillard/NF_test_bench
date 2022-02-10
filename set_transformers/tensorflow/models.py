from typing import Dict, List

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense
)

from .layers import (
    Encoder,
    Decoder
)


class SetTransformer(tf.keras.Model):

    def __init__(
        self,
        attention_axes: List[int],
        embedding_size: int,
        encoder_kwargs: Dict,
        decoder_kwargs: Dict,
        **kwargs
    ):
        """Fully parametrized Set Transformer
        SetTransformer(X) = Decoder(Encoder(Embedder(X)))

        Parameters
        ----------
        attention_axes : List[int]
            determines the axes on which attention is performed
        embedding_size : int
            determines dimension d across whole model
        encoder_kwargs : Dict
            Encoder kwargs
        decoder_kwargs : Dict
            Decoder kwargs
        """
        super().__init__(**kwargs)

        self.embedding_size = embedding_size
        self.attention_axes = attention_axes

        self.embedder = Dense(
            units=embedding_size,
            activation="relu"
        )

        self.encoder = Encoder(
            attention_axes=attention_axes,
            **encoder_kwargs
        )

        self.decoder = Decoder(
            attention_axes=attention_axes,
            **decoder_kwargs
        )

    def call(
        self,
        x: tf.Tensor
    ) -> tf.Tensor:
        """SetTransformer(X) = Decoder(Encoder(Embedder(X)))

        Parameters
        ----------
        x : tf.Tensor
            shape (batch_size, n, d')

        Returns
        -------
        tf.Tensor
            shape (batch_size, k, d)
        """
        e = self.embedder(x)
        z = self.encoder(e)

        return self.decoder(z)
