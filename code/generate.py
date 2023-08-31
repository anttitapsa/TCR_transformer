import os
import re
import torch
#import pickle
import numba
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm 
from transformers import AutoTokenizer 

from process_data import load_allele_datadict

class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, data, cdr3_length_distribution, tokenizer, CDR3_max_length):
        
        self.cdr3_length_distribution = cdr3_length_distribution
        self.tokenizer = tokenizer
        self.CDR3_max_length = CDR3_max_length
        self.V, self.J = fetch_sequences(data)
        self.V_codes, self.J_codes = fetch_sequence_codes(data)
        self.device = device
        
        if len(self.V) == 1:
            self.V_max_length = len(V[0].split()) + 1 #[CLS] token
            self.J_max_length = len(J[0].split()) + 2 #[SEP] token and one [PAD]
        else:

            V_length = 0
            J_length = 0
            for i in range(len(self.V)):
                #print(self.V[i])
                V_length = max(V_length, len(self.V[i].split()) + 1) #[CLS] token
                J_length = max(J_length, len(self.J[i].split()) + 2) #[SEP] token and one [PAD]
            
            self.V_max_length = V_length 
            self.J_max_length =  J_length 
            self.max_length = V_length + J_length + CDR3_max_length 
 
    def __len__(self):
        return len(self.V)

    def __getitem__(self, idx):
        
        cdr3_length = self.sample_length(self.V_codes[idx], self.J_codes[idx])
        CDR3 = cdr3_length * "[UNK]"
        seq = self.V[idx] + " " + CDR3 + " " + self.J[idx]
        
        tokenized = self.tokenizer(seq, padding= 'max_length', max_length= self.max_length, return_tensors='pt', return_attention_mask=True)
        
        padded_tokenized_seq = tokenized['input_ids']
        pad_mask = ~tokenized['attention_mask'].to(torch.bool)
        
        
        v_len = len(self.V[idx].split()) + 1 # Tokenizer adds [CLS] token at the beginning of the sequence
        lengths = torch.tensor([v_len, cdr3_length])
        
        V = padded_tokenized_seq[:, :v_len]
        J = padded_tokenized_seq[:, v_len+cdr3_length:]
        CDR3 = torch.zeros((cdr3_length))
        
        item = {'v': V.squeeze(0),
                'j': J.squeeze(0),
                'cdr3': CDR3,
                'length': lengths,
                'pad_mask': pad_mask.squeeze(0)}
        
        return item
    
    def sample_length(self,V, J):
        genes = (V, J)
        weights = torch.tensor(list(self.cdr3_length_distribution[genes].values()))
        lengths = torch.tensor(list(self.cdr3_length_distribution[genes].keys()))
        return int(lengths[int(torch.multinomial(input=weights, num_samples=1,replacement=True))])
    

@numba.jit(forceobj=True)
def fetch_sequences(data):
    allele_dict = load_allele_datadict(data_path) #remember to replace path
    for k, v in allele_dict.items():
        allele_dict[k] = " ".join(v[0].replace("-", "").replace(".",""))
        
    Vs, Js = [None]*data.shape[0], [None]*data.shape[0]
    for i in tqdm(range(data.shape[0]), position = 0, leave = True):
        v = f'{data[i, 3]}*0{int(data[i, 4])}'
        seq_v = allele_dict[v]
        Vs[i] = seq_v[:seq_v.rfind('C')+1]
        
        # The last element of the cdr3 is the F in 'combining region'
        # https://www.imgt.org/IMGTrepertoire/Proteins/alleles/index.php?species=Homo%20sapiens&group=TRBJ&gene=TRBJ-overview
        
        pattern = r'F G [A-Z] G'
        j = f'{data[i, 6]}*0{int(data[i, 7])}'
        seq_j = allele_dict[j]
        match = re.search(pattern, seq_j)
        Js[i] = seq_j[match.start()+1:] # F included in cdr3 
    
    return  Vs, Js

@numba.jit(forceobj=True)
def fetch_sequence_codes(data):
    V_codes, J_codes = [None]*data.shape[0], [None]*data.shape[0]
    for i in tqdm(range(data.shape[0]), position = 0, leave = True):
        V_codes[i] = f'{data[i, 3]}*{int(data[i, 4])}'
        J_codes[i] = f'{data[i, 6]}*{int(data[i, 7])}'
        
    return V_codes, J_codes

@numba.jit(forceobj=True)
def gene_combination_cdr3(data):
    result = {}
    for i in tqdm(range(data.shape[0]), position = 0, leave = True):
        v = f'{data[i, 3]}*{int(data[i, 4])}'
        j = f'{data[i, 6]}*{int(data[i, 7])}'
        cdr3_len = len(data[i,1])
        if (v, j) not in result.keys():
            result[(v,j)] = [cdr3_len]
        else:
            result[(v,j)].append(cdr3_len)

    return result

def collate(list_of_samples):
        result = {}
        for sample in list_of_samples:
            for k, v in sample.items():
                if k not in result.keys():
                    result[k] = [v]

                else:
                    result[k].append(v)
                    
        for k, v in result.items():
            if k == 'v' or k == 'j' or k =='cdr3':
                continue
            result[k] = torch.stack(v)
        return result

#################################################################

def multinomial_sample(model, Vs, Js, CDRs, pad_mask, length, device = 'cpu'):
    with torch.no_grad():
        prob = []
        cdr3_length = model.CDR3_max_length
        
        for i in tqdm(range(cdr3_length), leave=True, position=0):
            print(f'generating residue {i}')
            srcs = [torch.cat([v, cdr,j], axis=-1) for v, cdr, j in zip(Vs,Js,CDRs)]
            #print(srcs[0].shape)
            src = torch.stack(srcs).to(torch.int)
            #print(src)
            out = model(src=src.to(device),
                       length = length.to(device),
                       pad_mask =pad_mask.to(device),
                       device = device)
            #print(torch.multinomial(out.softmax(-1)[:,i], num_samples=1))
            for j in range(len(CDRs)):
                if i > CDRs[j].shape[-1]-1:
                    continue
                CDRs[j][i]= torch.multinomial(out.softmax(-1)[j,i], num_samples=1)
            #prob.append(out.softmax(-1)[:,i,:])
    return CDRs

##################################################################

if __name__ =='__main__':
    torch.set_default_dtype(torch.float64)
    data_path = os.path.join(os.getcwd(), os.pardir, os.pardir, "data")
    emerson_path =os.path.join(data_path, "emerson", "emerson_processed")
    test_data = pd.read_csv(os.path.join(emerson_path, "whole_seqs_nn_test.tsv"), sep = '\t')

    gene_cdr3 = gene_combination_cdr3(test_data.to_numpy())

    for i in tqdm(gene_cdr3.keys(), total = len(gene_cdr3), position = 0, leave = True):
        cdr3_count = {}
        total = 0
        lengths = gene_cdr3[i]
        lengths.sort()
        for j in lengths:
            key = j
            if key not in cdr3_count.keys():
                cdr3_count[key] = 1
            else:
                cdr3_count[key] += 1
            total += 1
        cdr3_count.update((x, y/total) for x, y in cdr3_count.items())
        gene_cdr3[i] = cdr3_count

    tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', add_special_tokens=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint =  torch.load("030823_customprotBERT_parallel_checkpoint.pth",  map_location="cpu")
    model = checkpoint["model"]
    model = model.module
    model.to(device)
    model.eval()

    test_set = GeneratorDataset(test_data.iloc[:2000].to_numpy(), gene_cdr3, tokenizer, model.CDR3_max_length)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=32, collate_fn=collate) 
    print('\nSTARTING TO GENERATE\n')
    for batch in test_loader:
        cdr3_pred = multinomial_sample(model, batch['v'], batch['j'], batch['cdr3'], batch['pad_mask'], batch['length'], device)
        print(cdr3_pred)
        