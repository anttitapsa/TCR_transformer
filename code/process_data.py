import os
import pandas as pd
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
    
    for i in range(len(data)):
        v_gene_family = data[i,1]
        j_gene_family = data[i,4]

        new_v_code = v_gene_family + '*0' + str(int(data[i,2]))
        data[i,1] = new_v_code

        new_j_code = j_gene_family + '*0' + str(int(data[i,5]))
        data[i,4] = new_j_code
    
    return data

@numba.jit(forceobj=True)
def replace_v_j(data, dictionary):
    
    for i in range(len(data)):
        data[i, 5] = ''.join(dictionary[data[i, 1]])
        data[i, 6] = ''.join(dictionary[data[i, 3]])
    return data

@numba.jit(forceobj=True)
def create_dataset(data):    
    V_data, J_data, CDR3_data, tgt_data = [None]*data.shape[0], [None]*data.shape[0], [None]*data.shape[0], [None]*data.shape[0]
    for i in range(data.shape[0]):
        V = data[i, 5].replace('-', '').replace('.', '')
        idx =V[:-math.ceil(data[i, 2]/3)-1].rfind('C')
        V = V[:idx+1]
        
        CDR3 = data[i, 0].replace('.', '')[1:]
        tgt = V + CDR3
        J = data[i, 6].replace('-', '')[math.ceil(data[i, 4]/3):].replace('.', '')
        
        idx2 = len(tgt) - tgt.rfind(J[0])
        if tgt[-idx2:] == J[:idx2]:
            J = J[idx2:]
            tgt = tgt +J
        else:
            tgt = tgt + J
        V_data[i] = " ".join(V)
        J_data[i] = " ".join(J)
        CDR3_data[i] = " ".join(CDR3)
        tgt_data[i] = " ".join(tgt)
    return V_data, CDR3_data, J_data,tgt_data

def process_data(train = True):
    data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "data"))
    emerson_data_path = os.path.join(data_path, "emerson", "emerson_processed")

    tsv_file = "whole_seqs_nn_train.tsv" if train else "whole_seqs_nn_test.tsv"
    
    print("Reading the .tsv file...")
    data_df = pd.read_csv(os.path.join(emerson_data_path, tsv_file), sep = '\t')
    data_df = data_df[['seq', 'v', 'v_allele', 'v_deletions', 'j', 'j_allele', 'j_deletions']]
    print("Creating the allele codes...")
    data_df = fix_gene_codes(data_df.to_numpy())
    
    data_df = pd.DataFrame(data=data_df, columns=['seq', 'v', 'v_allele', 'v_deletions', 'j', 'j_allele', 'j_deletions'])
    data_df['v_seq'] = ''
    data_df['j_seq'] = ''
    data_df = data_df.drop(labels=['v_allele','j_allele'], axis=1)
    
    print("Loading the sequences...")
    allele_seqs = load_allele_datadict(data_path=data_path)
    data = replace_v_j(data_df.to_numpy(), allele_seqs)
    
    print("Masking the dataset...")
    V_data, CDR3_data, J_data,tgt_data = create_dataset(data)

    return V_data, CDR3_data, J_data,tgt_data