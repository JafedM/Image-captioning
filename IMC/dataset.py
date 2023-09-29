import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

import os
import json

import numpy as np
from PIL import Image
from random import choice
from nltk.tokenize import TweetTokenizer
import argparse


def read_w2v(w2v_path, max_vocab=5000):
    w2v = []
    word_idx = {}
    idx_word = []
    with open(w2v_path, 'r') as file:
        for n, line in enumerate(file):
            if n==0:
                vocab_len, w2v_dim = line.split(' ')
            elif n==max_vocab:
                break
            else:
                line = line.split(' ')
                word_idx[line[0]] = n-1
                idx_word.append(line[0])
                w2v.append(list(map(float, line[1:])))
    return w2v, word_idx, idx_word

def load_captions_to_w2v(captions_json, imgs_path, max_n_caption=5, max_cap_len=15, tk=TweetTokenizer(), word_idx=None):
    '''
    captions_json:  Path a las anotaciones
    imgs_path:      Path a las imagenes
    max_n_caption:  Maxima cantidad de captions por imagen
    '''
    ann_dict = {} 
    #Revisando que imagenes estan realmente
    imgs = { img_file for img_file in os.listdir(imgs_path) }
    
    #revisando las captions
    with open(captions_json) as file:
        data = json.load(file)
        for img_json in data["images"]:
            '''
            Algunas imagenes estan en el json pero no en la carpeta por lo tanto 
            es mejor anotar las que si estan y las que no
            '''
            if img_json["file_name"] in imgs: 
                ann_dict[img_json["id"]] = { 
                    "file_name" : img_json["file_name"]
                 } 
                 
        for cap_info in data["annotations"]:
            img_id = cap_info["image_id"]
            if img_id in ann_dict:  #Aqui igual asegurar que la imagen este en los 2 lugares
                #Tokenizar caption
                cap = tk.tokenize(cap_info["caption"])

                #Ajustar len de caption
                if len(cap) < max_cap_len:
                    cap += ['</s>']*(max_cap_len-len(cap))
                elif len(cap) > max_cap_len:
                    cap = cap[:max_cap_len]

                if word_idx is not None:
                    cap = [word_idx[word.lower()] if word.lower() in word_idx else 0 for word in cap  ]
                
                # Cada imagen tiene varios captions
                if "caption" in ann_dict[img_id]:
                    #Agregar caption
                    if len(ann_dict[img_id]["caption"]) <  max_n_caption:
                        ann_dict[img_id]["caption"].append(cap)
                else:
                    ann_dict[img_id]["caption"] =[cap]
                    
    return ann_dict

class ICDataset(Dataset):
    def __init__(self, ann_dict, imgs_path, img_size=256):
        '''
        ann_dict:   Diccionario con las anotaciones ya tokenizadas y con embedding
        imgs_path:
        max_len:    Maxima longitud del caption
        words_embedding_idx
        '''
        self.ann_dict   = ann_dict
        self.imgs_path  = imgs_path

        self.img_to_tensor = transforms.functional.pil_to_tensor
        self.resize_img = transforms.Resize((img_size,img_size),antialias=True)

        #Al leer los caption se guarda el id que no esta numerado por lo tanto creo un vector de idx
        self.ann_idx = [x for x in self.ann_dict.keys()]

    def __len__(self):
        return len(self.ann_dict)
    
    def __getitem__(self, i):
        #Cargando imagen
        file_name = self.ann_dict[self.ann_idx[i]]["file_name"]
        img = Image.open(self.imgs_path + "/" + file_name)
        img = img.convert('RGB')#Algunas imagenes estan en blanco y negro
        img = self.resize_img(self.img_to_tensor(img)).float()

        #Revisar porque al parecer los esta inviertiendo
        img[0], img[1], img[2] = 255-img[0], 255-img[1], 255-img[2]
        
        #Cargando captions
        capts = self.ann_dict[self.ann_idx[i]]["caption"]
        capts = torch.tensor(capts[0])
        return img, capts