from re import T
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import random
import math
import time

import pickle

from torch.autograd import Variable
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        #self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(input_dim, enc_hid_dim, bidirectional = False)
        self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        #src = [src len, batch size]
        #embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]
        expand_temp = src
        outputs, hidden = self.rnn(expand_temp)

        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        #hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        hidden = torch.tanh(self.fc(hidden[-1,:,:]))
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #energy = [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2)
        #attention= [batch size, src len]
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        #self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(enc_hid_dim + 1000, dec_hid_dim)
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + 1000, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        input = input.unsqueeze(0)
        #input = [1, batch size]
        #embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]

        expand_output = input
        a = self.attention(hidden, encoder_outputs)
        #a = [batch size, src len]
        a = a.unsqueeze(1)
        #a = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        #weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        #weighted = [1, batch size, enc hid dim * 2]
        rnn_input = torch.cat((expand_output, weighted), dim = 2)
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        expand_output = expand_output.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, expand_output), dim = 1))
        #prediction = [batch size, output dim]
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
        input = torch.zeros(batch_size).to(self.device)
        for t in range(0, trg_len):
            
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            input = output
        outputs = torch.flip(outputs, dims = [0])
        return outputs, encoder_outputs

def train(model, iterator, optimizer, criterion, clip, device = None):
    
    model.train()
    
    epoch_loss = 0
    
    all_input = []
    all_encoder_output = []
    all_decoder_output = []
    for i, batch in enumerate(iterator):

        src = batch
        trg = batch
        src = np.expand_dims(src, axis = 1)
        trg = np.expand_dims(src, axis = 1)
        print('trg shape is')
        print(trg.shape)
        optimizer.zero_grad()
        
        output, encoder_outputs = model(Variable(torch.from_numpy(src).float()).to(device), Variable(torch.from_numpy(trg).float()).to(device))
        
        all_input.append(torch.from_numpy(src))
        all_encoder_output.append(encoder_outputs)
        all_decoder_output.append(output)

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]
        
        output = output.view(-1, output_dim)
        trg = torch.from_numpy(trg).to(device).view(-1, output_dim)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        if i % 100 == 0:
          print('current item is ' + str(i))
          print('current loss is')
          print(loss.item())
          '''
          print('current input is')
          print(src)
          print('current output is')
          print(output)
          print('current encoder output is')
          print(encoder_outputs)
          '''
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator), all_input, all_encoder_output, all_decoder_output

def evaluate(model, iterator, criterion, device = None):
    
    model.eval()
    
    epoch_loss = 0

    all_input = []
    all_encoder_output = []
    all_decoder_output = []

    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch
            trg = batch

            #src = np.expand_dims(src, axis = 1)
            #trg = np.expand_dims(src, axis = 1)

            output, encoder_outputs = model(Variable(torch.from_numpy(src).float()).to(device), 
        Variable(torch.from_numpy(trg).float()).to(device))

            all_input.append(torch.from_numpy(src))
            all_encoder_output.append(encoder_outputs)
            all_decoder_output.append(output)
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output.view(-1, output_dim)
            trg = torch.from_numpy(trg).to(device).view(-1, output_dim)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator), all_input, all_encoder_output, all_decoder_output

def get_distance(list1, list2, loss):
    total_distance = 0
    for item1 in list1:
        for item2 in list2:
            total_distance += loss(item1, item2).item()
    
    return total_distance/(len(list1) * len(list2))

def random_choose(input_tensor, chosen_parameters):
    def th_delete(tensor, indices):
        mask = torch.zeros(tensor.numel(), dtype=torch.bool)
        for i in indices:
            mask[i] = True
        return tensor[mask]
    return th_delete(input_tensor, chosen_parameters)

def main():
    chosen_parameter = np.random.choice(998, 50, replace = False)
    with open("src/train_gradient.pl", "rb") as fp:   # Unpickling
        raw = pickle.load(fp)
    
    input_data = []
    for agent in raw:
      temp_list = []
      for series_item in agent:
        temp_list.append(series_item.cpu().detach().numpy())
      input_data.append(temp_list)
    random.shuffle(input_data)
    train_data = np.array(input_data[5:])
    print('shape of train_data is')
    print(train_data.shape)

    val_data = np.array(input_data[0:5])
    print('shape of val_data is')
    print(val_data.shape)

    with open("src/corrupt_gradient.pl", "rb") as fp:   # Unpickling
        raw_corrupt = pickle.load(fp)
    
    corrupt_data = []
    for agent in raw_corrupt:
      temp_list = []
      for series_item in agent:
        temp_list.append(series_item.cpu().detach().numpy())
      corrupt_data.append(temp_list)

    corrupt_data = np.array(corrupt_data)
    print('shape of corrupt_data is')
    print(corrupt_data.shape)

    para_length = 1000
    INPUT_DIM = 1000
    OUTPUT_DIM = 1000
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 32
    DEC_HID_DIM = 32
    ENC_DROPOUT = 0
    DEC_DROPOUT = 0
    N_EPOCHS = 10
    CLIP = 1

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(enc, dec, device).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    pipeline = Pipeline([('normalizer', Normalizer()),
                        ('scaler', MinMaxScaler())])

    #pipeline.fit(train_data)

    #train_data = pipeline.transform(train_data)
    #val_data = pipeline.transform(val_data)
    #corrupt_data = pipeline.transform(corrupt_data)

    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
                
    model.apply(init_weights)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs





    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        train_loss, train_input, train_encoder_output, train_decoder_output = train(model, train_data, optimizer, criterion, CLIP, device)
        valid_loss, val_input, val_encoder_output, val_decoder_output = evaluate(model, val_data, criterion, device)
        corrupt_loss, corr_input, corr_encoder_output, corr_decoder_output = evaluate(model, corrupt_data, criterion, device)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        '''
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut3-model.pt')
        '''
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.10f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.10f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print(f'\t Corrupt. Loss: {corrupt_loss:.10f} |  Val. PPL: {math.exp(corrupt_loss):7.3f}')

        criterion_compare = nn.CosineSimilarity(dim = 0)
        print('distance between train input & val input:')
        print(get_distance(train_input, val_input, criterion))

        print('distance between train input & corrupt input:')
        print(get_distance(train_input, corr_input, criterion))

        print('distance between train encoder output & val encoder output:')
        print(get_distance(train_encoder_output, val_encoder_output, criterion))

        print('distance between train encoder output & corr encoder output:')
        print(get_distance(train_encoder_output, corr_encoder_output, criterion))

        print('distance between train decoder output & val decoder output:')
        print(get_distance(train_decoder_output, val_decoder_output, criterion))

        print('distance between train decoder output & corr decoder output:')
        print(get_distance(train_decoder_output, corr_decoder_output, criterion))

main()