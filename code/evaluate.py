import os 
import pandas as pd 
import numpy as np
from tqdm import tqdm 
#import scipy.stats as stats
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer 
import pickle

from CDR3Dataset import CDR3Dataset

from process_data import process_data

def sampling(model, src, length, cdr, pad_mask, device):

    cdr_logits = model(src=src,
                       length=length,
                       pad_mask=pad_mask,
                       device=device)
    probabilities = cdr_logits.log_softmax(-1)
    
    cdr = torch.nn.functional.pad(input=cdr, pad=(0, model.CDR3_max_length -cdr.shape[-1]))

    replace_tensor = torch.ones(probabilities.shape[0], model.CDR3_max_length, 30).to(device)
    mask = torch.where(cdr>0, torch.tensor(1), cdr).unsqueeze(-1).expand(probabilities.shape).to(device)
    replace_tensor = replace_tensor*mask

    target_onehot = torch.zeros(probabilities.shape).to(device)
    target_onehot.scatter_(dim=2, index = cdr.view([src.shape[0], -1, 1]) ,src=replace_tensor)
    
    seq_prob = target_onehot * probabilities
    seq_prob = torch.exp(seq_prob.sum(-1).sum(-1))
    
    return seq_prob

def main():
    torch.set_default_dtype(torch.float64)

    print("Loading the test data...")
    V_data, CDR3_data, J_data, tgt_data = process_data(train=False)
    print("Test data loaded.\n")

    print("Prepearing the data for evaluation...")
    counts = {}
    for v, cdr, seq in zip(V_data, CDR3_data,tgt_data):

        if  'C ' + cdr not in counts.keys():
            counts['C ' +cdr] = [1,[seq], [(len(v)+1, len(cdr))]]
        else:
            counts['C ' +cdr][0] += 1
            counts['C ' +cdr][1].append(seq)
            counts['C ' +cdr][2].append((len(v)+1, len(cdr)))

    c_data_, seqs_, lengths_ = [], [], []


    for v in counts.values():
        c_data_.append(v[0])
        seqs_.append(v[1])
        lengths_.append(v[2]) 


    cdr3_seqs_ = list(counts.keys())
    c_data, cdr3_seqs, seqs, lengths = [], [], [], []

    for i in range(len(seqs_)):  #only need seqs that has appearance > 2 ??why?? --> No reason found for this but done since it is done in GRU paper
        if c_data_[i] > 2:
            c_data.append(c_data_[i])
            cdr3_seqs.append(cdr3_seqs_[i])
            seqs.append(seqs_[i])
            lengths.append(lengths_[i])

    cdr3_test = [ s[1:] for s, c in zip(cdr3_seqs, c_data) for _ in range(c)]

    lengths_test = []
    tgt_test = [] 
    for s, l in zip(seqs, lengths):
        tgt_test += s
        lengths_test += l

    v_test = []
    j_test = []
    for seq, lens in zip(tgt_test, lengths_test):
        v_test.append(seq[:lens[0]-1])
        j_test.append(seq[lens[0]+lens[1]+1:]) 
    print("Data is now ready for evaluation.\n")


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Loading the model..")
    checkpoint =  torch.load("030823_customprotBERT_parallel_checkpoint.pth",  map_location=device)
    model = checkpoint["model"]
    model.to(device)
    model.eval()
    print("Model loaded.\n")


    tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', add_special_tokens=False)

    print("Preparing the dataloader...")
    dataset = CDR3Dataset(V = v_test, CDR3 = cdr3_test, J = j_test, tgt = tgt_test, tokenizer=tokenizer, evaluate=True)
    test_loader = DataLoader(dataset, shuffle=False, batch_size=1000)
    print("Dataloader is ready!\n")

    print("Starting to count CDR3 sequence probabilities (P_infer)...")
    seq_probs = []

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'\nEvaluating', position=0):
            
            batch = {k:v.to(device) for k,v in batch.items()}
            
            probs = sampling(model = model,
                                src = batch['seq'],
                                length = batch['length'],
                                cdr = batch['CDR3_label'],
                                pad_mask = batch['pad_mask'],
                                device=device)
            seq_probs.append(probs.to('cpu'))
            with open('P_infer_CDR3_seqs_checkpoint.pkl', 'wb') as f:
                pickle.dump(seq_probs, f)
            f.close()
            print(idx)
    seq_probs = torch.cat(seq_probs)

    print("\nCalculating average probabilities...")
    cdr3_probs ={}
    for i, cdr in enumerate(cdr3_test):
        if cdr not in cdr3_probs.keys():
            cdr3_probs[cdr] = [float(seq_probs[i])]
        else:
            cdr3_probs[cdr].append(float(seq_probs[i]))   

    for k, v in cdr3_probs.items():
        cdr3_probs[k] = sum(v)/len(v)

    print("Saving the P_infer...")
    with open('P_infer_CDR3_seqs.pkl', 'wb') as f:
        pickle.dump(cdr3_probs, f)
    f.close()
    print("saving was successfull!")

if __name__ == '__main__':
    main()