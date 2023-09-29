import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttn(nn.Module):
    '''
    Cabeza de atention (misma notacion que en attention is all you need)
    '''

    def __init__(self, h, d_model):
        '''
        h: Cantidad de Cabezas de atention
        d_model: Tamano del vector que se maneja
        '''
        super().__init__()
        #Comprobar que se puede hacer la proyeccion
        assert d_model % h == 0, "d_model debe ser divisible por h"

        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h #Tomar que d_k = d_v 

        #Capas para el modelo de attention
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output = nn.Linear(d_model, d_model)
        #Atencion
        self.attention = Attention()
        #Normalizacion para la residual
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, values, mask=None):
        '''
        q: Querry Matrix
        k: Key Matrix
        v: Value Matrix
        mask: Si se hace la mascara o no
        '''
        #Guardar residual
        residual = query #Podria ser cualquiera porque se pasan copias del mismo

        #Tamano del batch para no perderlo al multiplicar
        batch_size = values.size(0)

        #Pasar los q,k,v por las lineales
        query, key, values = [layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                              for layer, x in zip(self.linears,(query, key, values))]
        
        #Pasarlos por la capa de attencion
        if mask is not None:
            mask = mask.unsqueeze(1)    #Hacer que los valores ajuste en el tamano
        x = self.attention(query, key, values, mask)

        #Concatenar y que quede del tamano orignial (batch_size, max_len_dim, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = self.output(x)
        
        #Agregar parte residual y normalizar
        x += residual
        return self.norm(x)



class Attention(nn.Module):
    '''
    Atencion querr, key, value

    Solo creo el forward para que se considere el gradiente y no tener copias de q,k,v
    '''

    def forward(self, query, key, values, mask=None):
        '''
        q: Querry Matrix
        k: Key Matrix
        v: Value Matrix
        '''
        #Se calculan QK^T/sqrt(d_k)
        attn_scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(query.size(-1))
        
        #Se agrega la mascara si se necesita
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
        #Se calcula softmax y multiplica: softmax(QK^T/sqrt(d_k))V
        attn_scores = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_scores, values)

