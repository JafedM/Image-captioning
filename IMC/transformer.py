import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from IMC.encoderdecoder import *

class Transformer(nn.Module):
    ''' 
    Implementacion de transformer con imagen

    vocab_len:  Longitud del vocabulario
    emb_dim:    Dimension del embedding
    d_model:    Tamano de lo tensores en el modelo (Como en el paper)
    d_hidden:   Tamano capas ocultas en feeed--forward
    n_layers=6: Capas del Decoder
    h:          Cabezas de atencion
    n_position: Positional encoding
    w2v:        Embedding w2v
    '''

    def __init__(self, vocab_len, emb_dim=512, d_model=512, d_hidden=2048,
                    n_layers=6, h=8, n_position=200, w2v=None):

        super().__init__()

        #Tamano de lo vectores en modelo
        self.d_model = d_model

        #Encoder de images
        self.encoder = EncoderImg(d_model, d_hidden)

        #Decoder
        self.decoder = Decoder(emb_dim, n_position, vocab_len, n_layers, d_model, d_hidden, h, w2v)

        #Ultima capa lineal
        self.trg_word_prj = nn.Linear(d_model, vocab_len, bias=False)

        #Inicializacion dada
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == emb_dim, 'Usar mismo tamano de emb y de d_model'


    def forward(self, img, trg_seq):
        #Codificar la imagen
        enc_output = self.encoder(img)
        #Usar decoder
        dec_output = self.decoder(trg_seq, enc_output, False)
        #Pasar por final
        seq_logit = self.trg_word_prj(dec_output)

        return seq_logit


