import os
import pandas as pd
import random
from tqdm import tqdm
import numba 
import numpy as np
import math

def load_allele_datadict(data_path):
    trbvs = os.path.join(data_path, "trbvs_full.tsv")
    trbjs = os.path.join(data_path, "trbjs_full.tsv")

    trbvs_df = pd.read_csv(trbvs, sep = "\t")
    trbjs_df = pd.read_csv(trbjs, sep = "\t")

    trbvs_df.loc[:, "1":"109"] = trbvs_df.loc[:, "1":"109"].fillna("-") 
    trbjs_df.loc[:, "1":"17"] = trbjs_df.loc[:, "1":"17"].fillna("-")

    dictionary = {}

    for indx, row in trbjs_df.iterrows():
        key = row["Allele"]
        if key not in dictionary.keys():
            dictionary[key] = []
        seq = row.loc["1":"17"].to_string(header = False, index = False)
        seq = seq.replace('\n', '')
        dictionary[key].append(seq)

    for indx, row in trbvs_df.iterrows():
        key = row["Allele"]
        if key not in dictionary.keys():
            dictionary[key] = []
        seq = row.loc["1":"109"].to_string(header = False, index = False)
        seq = seq.replace('\n', '')
        dictionary[key].append(seq)

    return dictionary

@numba.jit(forceobj=True)
def fix_gene_codes(data):
    
    for i in tqdm(range(len(data)),position=0, leave=True):
        v_gene_family = data[i,1]
        j_gene_family = data[i,3]

        new_v_code = v_gene_family + '*0' + str(int(data[i,2]))
        data[i,1] = new_v_code

        new_j_code = j_gene_family + '*0' + str(int(data[i,4]))
        data[i,3] = new_j_code
    
    return data

@numba.jit(forceobj=True)
def replace_v_j(data, dictionary):
    '''
    replaces the v and j gene names with the amino acid sequence. 
    Data needs to be numpy array. If the variable rnd is False the
    1st available allele from the dictionary is chosen. Otherwise the
    allele is chosen randomly if multiple alleles are available.
    
    Dictionary need to be in form where the key is the name of the gene,
    and the value is list of tuples, where the 1st element of the tuple is
    the name of the allele and the 2nd element is the amino acid sequence of
    the allele.
    '''
    #print(data)

    for i in tqdm(range(len(data)),position=0, leave=True):
        data[i, 5] = ''.join(dictionary[data[i, 1]])
        data[i, 6] = ''.join(dictionary[data[i, 3]])
    return data

@numba.jit(forceobj=True)
def create_dataset(data):    
    train_data = []
    for i in tqdm(range(data.shape[0]), position = 0, leave = True):
        src = "[CLS] " + " ".join(data[i, 5].replace('-', '')) + len(data[i, 0])*" [MASK]" + " "+ " ".join(data[i, 6].replace('-', '')) + " [SEP]"
        trg = "[CLS] " + " ".join(data[i, 5].replace('-', '') + data[i, 0] + data[i, 6].replace('-', '')) + " [SEP]"
        train_data.append((src,trg))
    return train_data

def process_data(train = True):
    data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "data"))
    emerson_data_path = os.path.join(data_path, "emerson", "emerson_processed")

    tsv_file = "whole_seqs_nn_train.tsv" if train else "whole_seqs_nn_test.tsv"

    data_df = pd.read_csv(os.path.join(emerson_data_path, tsv_file), sep = '\t')
    data_df = data_df[['seq', 'v', 'v_allele', 'j', 'j_allele']]

    allele_seqs = load_allele_datadict(data_path=data_path)

    data = replace_v_j(data_df.to_numpy(), allele_seqs)

    dataset = create_dataset(data)

    return dataset