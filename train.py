
import argparse
import math
import time
from tqdm import tqdm
import numpy as np
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from IMC.attention import *
from IMC.encoderdecoder import *
from IMC.layers import *
from IMC.transformer import *
from IMC.dataset import *


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def get_trgts(trg, device='cuda'):
    trg, trg_seq = trg[:, :-1], trg[:, 1:]
    return trg.to(device), trg_seq.to(device)


def train_epoch(model, train_dataloader, optimizer, device):
    ''' Epoch operation in training phase'''

    model.to(device)
    model.train()

    for batch in tqdm(train_dataloader, 'Entrenando'):

        # prepare data
        img, caption_seq = batch
        img = img.to(device)
        caption_seq, trg_seq = get_trgts(caption_seq, device)

        # forward
        optimizer.zero_grad()
        pred = model(img, caption_seq)

        #print(pred.shape)
        #print(trg_seq.shape)
        '''
        Como el pred es de dimension 3 y el trg_seq es de dim 2
        ajustamos a que pred sea de dim 2 y el trg_seq de dim 1
        para poder calcular el cross entropy
        el contigous es necsario para ajustar el tamano como en la atencion
        '''
        pred = pred.view(-1, pred.size(2))
        trg_seq = trg_seq.contiguous().view(-1)

        # backward and update parameters
        loss = F.cross_entropy(pred, trg_seq) # ,gold Revisar esto
        loss.backward()
        optimizer.step()

    return loss


def eval_epoch(model, val_dataloader, device):
    ''' Epoch operation in evaluation phase '''

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_dataloader, 'Validacion'):

            # prepare data
            img, caption_seq = batch
            img = img.to(device)
            caption_seq, trg_seq = get_trgts(caption_seq, device)

            # forward
            pred = model(img, trg_seq)

            #ajustar tamano
            pred = pred.view(-1, pred.size(2))
            trg_seq = trg_seq.contiguous().view(-1)

            #losss
            loss = F.cross_entropy(pred, trg_seq) #Revisar esto

    return loss



def train(model, epoch, train_dataloader, val_dataloader, optimizer, device, output_dir='checkpoints'):
    ''' Start training '''

    def print_performances(estado, loss_val, start_time, lr):
        print('{estado:12} ---------- loss_val: {loss_val: 8.5f}, lr actual: {lr:8.5f}, '\
              'Tiempo: {tiempo:3.3f} min'.format(
                  estado=f"{estado}", loss_val=loss_val, tiempo=(time.time()-start_time)/60, lr=lr))

    valid_losses = []
    for epoch_i in range(epoch):
        print('\nEpoch: ', epoch_i)

        start = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        # Current learning rate
        lr = optimizer.param_groups[0]['lr']
        print_performances('Entrenamiento', train_loss, start, lr)

        start = time.time()
        valid_loss= eval_epoch(model, val_dataloader, device)
        print_performances('Validacion', valid_loss, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'model': model.state_dict()}

        model_name = 'model.chkpt'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if valid_loss <= min(valid_losses):
            torch.save(checkpoint, os.path.join(output_dir, model_name))
            print('Guardando modelo')

def main():
    
    w2v, word_idx, idx_word = read_w2v('word2vec_lim.txt')

    train_ims   = r'C:\Users\jafse\Documents\Maestria cimat\Proyecto tecnologico\YOLOv5-mask\data\coco2017\train2017'
    val_ims     = r'C:\Users\jafse\Documents\Maestria cimat\Proyecto tecnologico\YOLOv5-mask\data\coco2017\val2017'
    train_caps  = r'C:\Users\jafse\Documents\Maestria cimat\Proyecto tecnologico\YOLOv5-mask\data\coco2017\annotations_2\captions_train2017.json'
    val_caps    = r'C:\Users\jafse\Documents\Maestria cimat\Proyecto tecnologico\YOLOv5-mask\data\coco2017\annotations_2\captions_val2017.json'

    train_dict = load_captions_to_w2v(train_caps, train_ims, word_idx=word_idx)
    val_dict   = load_captions_to_w2v(val_caps, val_ims, word_idx=word_idx)

    train_dataset = ICDataset(train_dict, train_ims)
    val_dataset = ICDataset(val_dict, val_ims)

    lr = 0.0001
    epochs = 5
    device = torch.device('cuda')
    weight_decay = 0.0001
    beta1 = 0.9
    beta2 = 0.999
    chkp_path = 'best'

    batch_size = 128

    train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True, 
                                drop_last=True)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True)

    torch.cuda.empty_cache()

    model = Transformer(vocab_len=len(word_idx), emb_dim=100, d_model=100, d_hidden=1028,
                    n_layers=1, h=1, n_position=50, w2v=torch.tensor(w2v))
    
    optimzer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas = (beta1, beta2))

    train(model, epochs, train_dataloader, val_dataloader, optimzer, device, chkp_path)

if __name__ == "__main__":
  main()
