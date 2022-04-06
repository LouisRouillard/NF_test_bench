import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List

from layers import Encoder, Decoder

NORMALIZE = True
NUM_HEADS = 2
NUM_INDS = 8 # like embedding_size
NUM_SEEDS = 1

NUM_LAYERS = 2 # SAB or ISAB layers for encoder

ENC_ARGS = {
'type':'ISAB',
'kwargs_per_layer':[
    {'num_heads':NUM_HEADS, 'normalize':NORMALIZE, 'num_inds':NUM_INDS}
    ] * NUM_LAYERS
}

DEC_ARGS = {
'pma_kwargs':{'num_heads':NUM_HEADS, 'num_seeds':NUM_SEEDS,
    'normalize':NORMALIZE, 'rff':True}, # rff False in authors implementation
'sab_kwargs':{'num_heads':NUM_HEADS, 'normalize':NORMALIZE}
}

class SetTransformer(nn.Module):

    def __init__(
        self,
        in_features: int,
        embedding_size: int,
        out_features: int = None,
        encoder_kwargs: Dict = ENC_ARGS,
        decoder_kwargs: Dict = DEC_ARGS,
        **kwargs
    ):
        """ Fully parametrized Set Transformer
        SetTransformer(X) = Decoder(Encoder(Embedder(X)))

        Parameters
        ----------
        in_features: int
            Data point dimension (of elements of input set X)
        embedding_size: int
            Data point dimention after embedding of elements of X
            (`dim` in MAB/SAB/ISAB)
        out_features: int, optional
            Data point dimension of the output
            by default = None.
            if not None, a final Linear layer is applied
            (as in autjors implementation)
        encoder_kwargs: Dict
            Encoder kwargs (type, kwargs_per_layer)
            except dim that is defined in __init__()
        decoder_kwargs: Dict
            Decoder kwargs (pma_kwargs, sab_kwargs)
            except dim that is defined in __init__()
        """
        super(SetTransformer, self).__init__()

        self.embedding_size = embedding_size

        self.embedder = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.ReLU() # relu not used in authors embedding
        )

        encoder_kwargs['dim'] = embedding_size
        self.encoder = Encoder(**encoder_kwargs)

        decoder_kwargs['dim'] = embedding_size
        self.decoder = Decoder(**decoder_kwargs)

        if out_features is not None:
            self.fc_out = nn.Linear(embedding_size, out_features)
        else:
            self.fc_out = nn.Identity()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """SetTransformer(X) = Decoder(Encoder(Embedder(X)))

        Parameters
        ----------
        X : torch.Tensor
            shape (batch_size, n, in_features)

        Returns
        -------
        torch.Tensor
            shape (batch_size, num_seeds, out_features)
        """
        E = self.embedder(X) # batch_size, n, embedding_size
        Z = self.encoder(E) # batch_size, n, embedding_size
        O = self.decoder(Z) # batch_size, num_seeds, embedding_size
        return self.fc_out(O) # batch_size, num_seeds, out_features
