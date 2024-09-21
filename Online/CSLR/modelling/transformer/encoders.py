# coding: utf-8

import torch
import torch.nn as nn
from torch import Tensor
from modelling.transformer.layers import TransformerEncoderLayer, PositionalEncoding

# pylint: disable=abstract-method
class Encoder(nn.Module):
    """
    Base encoder class
    """

    @property
    def output_size(self):
        """
        Return the output size
        :return:
        """
        return self._output_size



class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        pe: bool = True,
        LN: bool = True,
        skip_connection: bool=True,
        output_size: int=512,
        **kwargs
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TransformerEncoder, self).__init__()
        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    fc_type=kwargs.get('fc_type', 'linear'),
                    kernel_size=kwargs.get('kernel_size', 1),
                    skip_connection=skip_connection
                )
                for _ in range(num_layers)
            ]
        )
        if LN:
            self.layer_norm = nn.LayerNorm(output_size, eps=1e-6)
        else:
            print('Turn off layer norm at the last of encoder')
            self.layer_norm = nn.Identity()
        if pe:
            self.pe = PositionalEncoding(hidden_size)
        else:
            print('Turn off positional encoding')
            self.pe = None
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = output_size
        if self._output_size != hidden_size:
            print('transformer outputsize {} != hidden size {}'.format(self._output_size, hidden_size))
            print('Create a mapping layer')
            self.map2gloss_embed = nn.Sequential(
                nn.Linear(hidden_size, self._output_size),
                nn.Dropout(dropout),
            )
        else:
            self.map2gloss_embed = nn.Identity()
        # if freeze:
        #     freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src, mask, output_attention=False
    ):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].
        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        intermediate = {}
        intermediate['sgn_embed'] = embed_src
        if self.pe:
            x = self.pe(embed_src)  # add position encoding to word embeddings
            intermediate['pe'] = x
        else:
            x = embed_src
        x = self.emb_dropout(x)
        if output_attention:
            attentions = []
        for li, layer in enumerate(self.layers):
            if output_attention:
                x, attention = layer(x, mask, output_attention=False)
                attentions.append(attention)
            else:
                x = layer(x, mask)
            intermediate['layer_'+str(li)] = x

        x = self.map2gloss_embed(x)
        x = self.layer_norm(x)
        intermediate['gloss_feature'] = x #B,T,D
        if output_attention:
            attentions = torch.stack(attentions, dim=1) #B, L, H, T,T
        else:
            attentions = None

        return x, None, attentions, intermediate # None -> encoder hidden(unused)

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].src_src_att.num_heads,
        )
