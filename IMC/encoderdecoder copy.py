import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
from IMC.attention import *


class Encoder(nn.Module):
    '''
    Encoder completo
    '''
    def __init__(self, emb_dim, n_position, vocab_len,
                 n_layers, d_model, d_hidden, h):
        '''
        dim_emmb: Dimension del emmbeding
        n_position: Posiciones para el embeding posi
        n_layers: Cantidad de capas de encoder
        h: Cantidad de Cabezas de atention
        d_model: Tamano del vector que se maneja
        d_hidden: Tamano coulto del feed-forward
        '''
        super(Encoder, self).__init__()
        #Embeding
        self.emb = nn.Embedding(vocab_len, emb_dim)
        #Positional emmb
        self.position_emb = PositionalEncoding(emb_dim, n_position=n_position)
        #Capas de encoder
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, d_hidden, h)
                                          for _ in range(n_layers)])
        #
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, src_seq, mask=False):
        #Pasar por el embeding
        encoder_out = self.emb(src_seq)
        #Pasar por el posemb
        encoder_out = self.position_emb(encoder_out)

        #Pasar por las capas de encoder
        for enc_layer in self.encoder_layers:
            encoder_out = enc_layer(encoder_out, mask=mask)

        return encoder_out
    
class Decoder(nn.Module):
    '''
    Encoder completo
    '''
    def __init__(self, emb_dim, n_position, vocab_len,
                 n_layers, d_model, d_hidden, h, w2v=None):
        '''
        dim_emmb: Dimension del emmbeding
        n_position: Posiciones para el embeding posi
        n_layers: Cantidad de capas de encoder
        h: Cantidad de Cabezas de atention
        d_model: Tamano del vector que se maneja
        d_hidden: Tamano coulto del feed-forward
        '''
        super(Decoder, self).__init__()
        #Embeding
        if w2v is not None:
            self.emb = nn.Embedding.from_pretrained(w2v)
        else:
            self.emb = nn.Embedding(vocab_len, emb_dim)
        #Positional emb
        self.position_emb = PositionalEncoding(emb_dim, n_position=n_position)
        #Capas de encoder
        self.encoder_layers = nn.ModuleList([DecoderLayer(d_model, d_hidden, h)
                                          for _ in range(n_layers)])
        #
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, trg_seq, encoder_out, mask=False):
        #Pasar por el embeding
        decoder_out = self.emb(trg_seq)
        #Pasar por el posemb
        decoder_out = self.position_emb(decoder_out)

        #Pasar por las capas de encoder
        for enc_layer in self.encoder_layers:
            decoder_out = enc_layer(decoder_out, encoder_out, mask=mask)

        return decoder_out

class EncoderLayer(nn.Module):
    '''
    Capa del encoder
    
    Esta parte hace todas las cuentas de la imagen del paper una sola vez
    '''
    def __init__(self, d_model, d_hidden, h):
        '''
        h: Cantidad de Cabezas de atention
        d_model: Tamano del vector que se maneja
        d_hidden: Tamano coulto del feed-forward
        '''
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttn(h, d_model) #El encoder solo tiene una capa de atencion
        self.pos_fedfod = PossFeedForward(d_model, d_hidden)

    def forward(self, x, mask=False):
        #Pasar por la cabeza de atencion
        out = self.attn(x, x, x, mask)

        #Pasar por el feed-forward
        return self.pos_fedfod(out)
    

class EncoderLayerImg(nn.Module):
    '''
    Capa del encoder
    Se usa una resnet, se hace flaten y se pasa por una lineal
    '''
    def __init__(self, d_model, d_hidden):
        '''
        h: Cantidad de Cabezas de atention
        d_model: Tamano del vector que se maneja
        d_hidden: Tamano coulto del feed-forward
        '''
        super(EncoderLayerImg, self).__init__()

        #Resnet50 No entrenable
        self.resnet = torchvision.models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        #salida tamano mil de resnet50
        flatten_dim = 1000 
        #Capas lineales
        self.w_1 = nn.Linear(flatten_dim, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        #Normalizacion para la residual
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, img):
        #Pasar por la cabeza de atencion
        out = self.resnet(img)
        #Pasar por el feed-forward
        out = self.w_2(F.relu(self.w_1(out)))
        return self.norm(out)


class DecoderLayer(nn.Module):
    '''
    Capa del encoder
    
    Esta parte hace todas las cuentas de la imagen del paper una sola vez
    '''
    def __init__(self, d_model, d_hidden, h):
        '''
        h: Cantidad de Cabezas de atention
        d_model: Tamano del vector que se maneja
        d_hidden: Tamano coulto del feed-forward
        '''
        super(DecoderLayer, self).__init__()
        #El decoder tiene 2 atenciones
        self.slf_attn = MultiHeadAttn(h, d_model)   #self attention
        self.cross_attn = MultiHeadAttn(h, d_model) #cross attention
        self.pos_fedfod = PossFeedForward(d_model, d_hidden)

    def forward(self, decoder_input, encoder_out, mask=False):
        #Pasar por auto atencion
        decoder_out = self.slf_attn(decoder_input, decoder_input, decoder_input, mask)

        #Cross atencion (querry del decoder y los demas del encoder)
        decoder_out = self.slf_attn(decoder_out, encoder_out, encoder_out, mask)

        #Pasar por el feed-forward
        return self.pos_fedfod(decoder_out)


class PossFeedForward(nn.Module):
    ''' 
    Hacer el feed-forward con normalizacion y recidual
    '''

    def __init__(self, d_model, d_hidden):
        super().__init__()
        #Capas lineales
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        #Normalizacion para la residual
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        #Guardar residual
        residual = x.clone()

        #Hacer el feed-forward
        x = self.w_2(F.relu(self.w_1(x)))
        
        #Agregar parte residual y normalizar
        x += residual
        return self.norm(x)
    
class PositionalEncoding(nn.Module):
    '''
    Extraido de una implementacion
    '''

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()