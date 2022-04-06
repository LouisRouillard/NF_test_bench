import torch
import torch.nn as nn
import math

from typing import List, Dict, Tuple

class MAB(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        normalize: bool = True,
        **kwargs
    ):
        """Multihead Attention Block:
        MAB(X, Y) = LayerNorm(H + rFF(H)) ∈ R n × dim
        where H = LayerNorm(X + Multihead(X, Y, Y))

        Parameters
        ----------
        dim: int
            Data point dimension (of the elements of X and Y)
        num_heads: int
            Number of heads in the multi-head rchitecture
        normalize: bool, optional
            if True, use LayerNorm layers as part of
            the architecture (as per the original paper),
            by default True
        """

        super(MAB, self).__init__()

        self.dim = dim
        # typical choice for the split dimension of the heads
        self.dim_split = dim // num_heads

        # # embeddings for multi-head projections
        # self.fc_x = nn.Linear(dim_in, dim_out)
        # self.fc_y_k = nn.Linear(dim_in, dim_out) #dim_x == dim_k
        # self.fc_y_v = nn.Linear(dim_in, dim_out) #dim_v == dim_out

        # row-wise feed-forward layer
        self.rff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

        self.normalize = normalize
        if normalize:
            self.layer_norm_h = nn.LayerNorm(dim)
            self.layer_norm_out = nn.LayerNorm(dim)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """ Compute MAB(X,Y)

        Parameters
        ----------
        X: torch.Tensor of size (batch_size, n, dim)
            Query: can be previously embedded to fit data-dimension dim
        Y: torch.Tensor of size (batch_size, m, dim)
            Key and Value: can be previously embedded to fit data-dimension dim

        Returns
        -------
        O: torch.Tensor of size (batch_size, n, dim)
            Output MAB(X,Y)
        """
        # # projection onto same dim
        # X = self.fc_x(X)
        # K, V = self.fc_y_k(Y), self.fc_y_v(Y)

        assert(X.shape[-1] == self.dim)
        assert(Y.shape[-1] == self.dim)

        # Split into num_head vectors (num_heads * batch_size, n/m, dim_split)
        Q = torch.cat(X.split(self.dim_split, 2), 0)
        K = torch.cat(Y.split(self.dim_split, 2), 0)
        V = torch.cat(Y.split(self.dim_split, 2), 0)

        # Attention weights of size (num_heads * batch_size, n, m):
        # measures how similar each pair of Q and K is.
        W = torch.softmax(Q.bmm(K.transpose(1,2))/math.sqrt(self.dim), 2)

        # Multihead output (batch_size, n, dim):
        # weighted sum of V where a value gets more weight if its corresponding
        # key has larger dot product with the query.
        H = torch.cat((Q + W.bmm(V)).split(X.size(0), 0), 2)
        if self.normalize:
            H = self.layer_norm_h(H)
        O = H + self.rff(H)
        if self.normalize:
            O = self.layer_norm_out(O)
        return O

class SAB(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        normalize: bool = True,
        **kwargs
    ):
        """Self Attention Block:
        SAB(X) = MAB(X, X) ∈ R n × dim

        Parameters
        ----------
        dim: int
            Data point dimension (of the elements of X)
        num_heads: int
            Number of heads in the multi-head architecture
        normalize: bool, optional
            if True, use LayerNorm layers as part of
            the architecture (as per the original paper),
            by default True
        """
        super(SAB, self).__init__()
        self.mab = MAB(dim, num_heads, normalize=normalize)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Compute SAB(X)

        Parameters
        ----------
        X: torch.Tensor of size (batch_size, n, dim)
            Query, Key and Value: can be previously embedded to fit
            data-dimension dim

        Returns
        -------
        Output SAB(X): torch.Tensor of size (batch_size, n, dim)
        """
        return self.mab(X,X)

class ISAB(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_inds: int,
        normalize: bool = True,
        **kwargs
    ):
        """Induced Self Attention Block:
        ISAB (X) = MAB(X, H) ∈ R n × dim
        where H = MAB(I, X) ∈ R num_inds × dim

        Parameters
        ----------
        dim: int
            Data point dimension (of the elements of X)
        num_heads: int
            Number of heads in the multi-head architecture
        num_inds: int
            Number of inducing points
        normalize: bool, optional
            if True, use LayerNorm layers as part of
            the architecture (as per the original paper),
            by default True
        """
        super(ISAB, self).__init__()

        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim))
        nn.init.xavier_uniform_(self.I)

        self.mab_h = MAB(dim, num_heads, normalize=normalize)
        self.mab_out = MAB(dim, num_heads, normalize=normalize)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Compute ISAB(X)

        Parameters
        ----------
        X: torch.Tensor of size (batch_size, n, dim)
            Query, Key and Value: can be previously embedded to fit
            data-dimension dim

        Returns
        -------
        Output ISAB(X): torch.Tensor of size (batch_size, n, dim)
        """

        H = self.mab_h(self.I.repeat(X.size(0),1,1), X)
        return self.mab_out(X, H)

class Encoder(nn.Module):
    def __init__(
        self,
        type: str,
        kwargs_per_layer: List[Dict],
        dim: int,
        **kwargs
    ):
        """Set Transformer encoder
        Stack of SAB or ISAB blocks
           Encoder(X) = SAB(SAB(... X)) ∈ R n × dim
        or Encoder(X) = ISAB(ISAB(... X)) ∈ R n × dim

        Parameters
        ----------
        type : str
            one of ["SAB", "ISAB"]
        kwargs_per_layer: List[Dict]
            kwargs for SAB or ISAB class (num_heads, normalize)
            except dim that has to be the same for all
        dim: int
            Data point dimension (of the elements of X)

        Raises
        ------
        AssertionError
            if type is not in ["SAB", "ISAB"]
        """
        super(Encoder, self).__init__()

        if type not in ["SAB", "ISAB"]:
            raise AssertionError(
                "type should be one of [`SAB`, `ISAB`]"
            )

        for layer_kwarg in kwargs_per_layer:
            layer_kwarg['dim'] = dim

        layers=[
            (SAB if type == "SAB" else ISAB)(**layer_kwargs)
            for layer_kwargs in kwargs_per_layer
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Encoding for Set Transformer (stack of SAB or ISAB layers)

        Parameters
        ----------
        X: torch.Tensor of size (batch_size, n, dim)

        Returns
        -------
        Encoding of size (batch_size, n, dim)
        """
        return self.seq(X)

class PMA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_seeds: int,
        normalize: bool = True,
        rFF: bool = False,
        **kwargs
    ):
        """Pooling by Multihead Attention block:
        PMA(Z) = MAB(S, rFF(Z)) ∈ R num_seeds × dim
        where S ∈ R num_seeds × dim

        Parameters
        ----------
        dim: int
            Data point dimension (of the elements of Z)
        num_seeds : int
            number of seed vectors
        normalize: bool, optional
            if True, use LayerNorm layers as part of
            the architecture (as per the original paper),
            by default True
        rFF: bool, optional
            if True use rFF to embedd Z (as in paper)
            by default False (as in authors implementation)
        """
        super(PMA, self).__init__()

        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)

        self.rff = nn.Identity()
        if rFF:
            self.rff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

        self.mab = MAB(dim, num_heads, normalize=normalize)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute PMA(Z)

        Parameters
        ----------
        Z: torch.Tensor of size (batch_size, n, dim)
            Output of the Encoder

        Returns
        -------
        Pooled data PMA(Z): torch.Tensor of size (batch_size, num_seeds, dim)
        """
        Z_ = self.rff(Z) # not in authors implementation

        return self.mab(self.S.repeat(Z.size(0), 1, 1), Z_)

class Decoder(nn.Module):
    def __init__(
        self,
        pma_kwargs: Dict, # num_heads, num_seeds, normalize
        sab_kwargs: Dict, # num_heads, normalize
        dim: int,
        **kwargs
    ):
        """Set Transformer Decoder
        Decoder(Z) = rFF(SAB(PMA(Z))) ∈ R num_seeds × dim

        Parameters
        ----------
        pma_kwargs : Dict
            PMA kwargs (num_heads, num_seeds, normalize, rff)
            except dim that has to be the same for all
        sab_kwargs : Dict
            SAB kwargs (num_heads, normalize)
            except dim that has to be the same for all
        dim : int
            Data point dimension (of the elements of Z)
        """
        super(Decoder, self).__init__()

        pma_kwargs['dim'] = dim
        self.pma = PMA(**pma_kwargs)

        sab_kwargs['dim'] = dim
        self.sab = SAB(**sab_kwargs)

        self.rff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Decoding for Set Transformer

        Parameters
        ----------
        Z: torch.Tensor of size (batch_size, n, dim)
            Output of the Encoder

        Returns
        -------
        Decoded data: torch.Tensor of size (batch_size, num_seeds, dim)
        """
        return self.rff(self.sab(self.pma(Z))) # only one sab not two?
