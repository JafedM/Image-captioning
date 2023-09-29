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
                    n_layers=6, h=8, n_position=200, w2v=None, mask=None):

        super().__init__()

        #Tamano de lo vectores en modelo
        self.d_model = d_model

        #Enmascarar caption
        self.mask = mask

        #Encoder de images
        self.encoder = EncoderImg(d_model, d_hidden)

        #Decoder
        if w2v is not None:
            self.emb = nn.Embedding.from_pretrained(w2v)
        else:
            self.emb = nn.Embedding(vocab_len, emb_dim)
        #Positional emb
        self.position_emb = PositionalEncoding(emb_dim, n_position=n_position)
        #self.decoder = Decoder(emb_dim, n_position, vocab_len, n_layers, d_model, d_hidden, h, w2v)
        self.decoder = nn.TransformerDecoderLayer(d_model, h, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder, n_layers)
        #Ultima capa lineal
        self.trg_word_prj = nn.Linear(d_model, vocab_len, bias=False)

        #Inicializacion dada
        #for p in self.parameters():
        #    if p.dim() > 1:
        #        nn.init.xavier_uniform_(p) 

        assert d_model == emb_dim, 'Usar mismo tamano de emb y de d_model'


    def forward(self, img, trg_seq):
        #Codificar la imagen
        enc_output = self.encoder(img)

        #Usar decoder
        if self.mask is not None:
            #self.mask = get_pad_mask(trg_seq, 0) & get_subsequent_mask(trg_seq)
            self.mask = get_mask(trg_seq.size(1)).to('cuda')

        #Pasar por el embeding
        trg_seq = self.emb(trg_seq)
        #Pasar por el posemb
        trg_seq = self.position_emb(trg_seq)
        #Decoder
        dec_output = self.decoder(trg_seq, enc_output, self.mask)
        #Pasar por final
        seq_logit = self.trg_word_prj(dec_output)

        return seq_logit

def get_mask(size):
    mask = torch.tril(torch.ones(size, size)) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
    return mask

def get_subsequent_mask(seq):
    ''' 
    Enmascarar todo lo de adelante
    '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1))
    return subsequent_mask

def get_pad_mask(seq, pad_idx):
    '''
    Enmascarar alguna palabra en especifico
    '''
    return (seq != pad_idx).unsqueeze(-2)