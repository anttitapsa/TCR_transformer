import pandas as pd
import numpy as np
import os
import math

'''
This script is modified version of Yuepeng Jiang and Shuai Cheng Li's data process script they utilised for training TCRpeg.
Original script can be found on https://github.com/jiangdada1221/TCRpeg/blob/main/tcrpeg/process_data.py and the TCRpeg article 
("Deep autoregressive generative models capture the intrinsics embedded in T-cell receptor repertoires") can be found on
 https://doi.org/10.1093/bib/bbad038.

This script creates dataset in .tsv format by filtering the Emerson et al TCR dataset which can be found on https://doi.org/10.1038/ng.3822

How to use this code for getting the universal TCR pool from Emerson et al.:
First put the 743 files of individual repertoires under the folder 'original_data/' and create folder called 'data/'.
After that run the program on the terminal using below prompt;
python3 process_data.py

After execution, it will generate three files 'whole_seqs_nn.tsv', 'whole_seqs_nn_train.tsv', and 'whole_seqs_nn_test.tsv' under the 'data/' folder.
'whole_seqs_nn.tsv' contains all of the datapoints, and in 'whole_seqs_nn_train.tsv' and 'whole_seqs_nn_test.tsv' the data is split to train and test set. 
During the split of the data set to train and test set, seed 42 is utilised, so that the same train and test set can be created again later.
'''


vs_default = [ 'TRBV10-1','TRBV10-2','TRBV10-3','TRBV11-1','TRBV11-2','TRBV11-3','TRBV12-5', 'TRBV13', 'TRBV14', 'TRBV15', 
               'TRBV16', 'TRBV18','TRBV19','TRBV2', 'TRBV20-1', 'TRBV25-1', 'TRBV27', 'TRBV28', 'TRBV29-1', 'TRBV3-1', 'TRBV30',
               'TRBV4-1', 'TRBV4-2','TRBV4-3', 'TRBV5-1', 'TRBV5-4', 'TRBV5-5', 'TRBV5-6', 'TRBV5-8','TRBV6-1', 'TRBV6-4',
               'TRBV6-5', 'TRBV6-6','TRBV6-8', 'TRBV6-9', 'TRBV7-2', 'TRBV7-3', 'TRBV7-4', 'TRBV7-6', 'TRBV7-7', 'TRBV7-8', 'TRBV7-9', 'TRBV9']
js_default = ['TRBJ1-1', 'TRBJ1-2', 'TRBJ1-3','TRBJ1-4', 'TRBJ1-5', 'TRBJ1-6','TRBJ2-1', 'TRBJ2-2', 'TRBJ2-3', 'TRBJ2-4', 'TRBJ2-6', 'TRBJ2-7']

def get_whole_data(files,aa_column_beta=52+1,v_beta_column=52+10,j_beta_column=52+16,frame_column=52+2,nc_column=52, v_allele_column = 52+11,
                    j_allele_column = 52+17, v_deletion_column=52+18, j_deletion_column=52+21, patient_id_column=0,
                    max_length=30,filter_genes=True, columns_by_name=False, train_frac = 0.5):
    
    '''
   Create dataset based on the list of the files on Emerson dataset. created .tsv files contains information following information:
   Patient ID, CDR3 amin acid sequence, CDR3 nucleotide sequence, and names, alleles, and deletions of the V and J genes. Default values of the 
   column indexes are based on the emerson dataset, but the names of the columns can be utilised instead of the indexes. In that case the variable 
   columns_by_name needs to be set True. Filtering happens based on the preprocessing steps of SoNNia and TCRvae. 

    param files: List of the paths of files to be processed.
    param aa_column_beta: The index of the column containing the CDR3 amino acid sequence. If columns_by_name is True,
                           the name of the column should be used instead of the index.
    param v_beta_column: The index of the code of the V gene.
    param j_beta_column: The index of the code of the J gene.
    param frame_column: The index of the column containing the frame information.
    param nc_column: The index of the column containing the nucleotide sequence of CDR3.
    param v_allele_column: The index of the column containing the V gene allele information.
    param j_allele_column: The index of the column containing the J gene allele information.
    param v_deletion_column: The index of the column containing the V gene deletion information.
    param j_deletion_column: The index of the column containing the J gene deletion information.
    param patient_id_column: The index of the column containing the patient ID information.
    param max_length: Maximum length of the sequence (default is 30).
    param filter_genes: Whether to filter genes that are not in vs_default and js_default. This step is just used to follow
                         soNNia and TCRvae since they both exclude genes that are not in the global variable lists
                         vs_default and js_default.
    param columns_by_name: Whether the column indices are specified by name or index.
    param train_frac: Fraction of data used for training (default is 0.5). remaining part of the data is utilised as a test set.

    '''

    print('loading Data')
    in_frame=True
    seps = ['\t' if f.endswith('.tsv') else ',' for f in files]
    unique_seqs = set() # to store nn now

    for i,file in enumerate(files):
        if i % 10 == 0:
            print('Have processed {} files of {} files. The current file is {}.'.format(i+1, len(files), file))
        
        beta_sequences = []
        v_beta,j_beta = [],[]
        nn_beta = []
        v_al_beta,j_al_beta = [],[]
        v_deletion, j_deletion = [], []
        patient_id = []
        f = pd.read_csv(file,sep=seps[i], low_memory=False)

        if not columns_by_name:        
            col_nn = f[f.columns[nc_column]].values
            col_v, col_j = f[f.columns[v_beta_column]].values, f[f.columns[j_beta_column]].values
            col_v,col_j = [ v if isinstance(v, float) and math.isnan(v) else v[0] + v[2:5] + v[6:] if v[5] == '0' else v[0] + v[2:] for v in col_v], [j if isinstance(j, float) and math.isnan(j) else j[0] + j[2:5] + j[6:] if j[5] == '0' else j[0] + j[2:]  for j in col_j]
            col_v,col_j = [v if isinstance(v, float) and math.isnan(v) else v[:-2] + v[-1] if v[-2] == '0' else v for v in col_v], [j if isinstance(j, float) and math.isnan(j) else j[:-2] +j[-1] if j[-2] == '0' else j for j in col_j]
            col_beta = f[f.columns[aa_column_beta]].values if in_frame else f[f.columns[nc_column]]
            frame_bool = f[f.columns[frame_column]].values 
            col_v_al,col_j_al = f[f.columns[v_allele_column]].values,f[f.columns[j_allele_column]].values
            col_v_del, col_j_del = f[f.columns[v_deletion_column]].values, f[f.columns[j_deletion_column]].values
            col_id = f[f.columns[patient_id_column]].values
       
        if columns_by_name:
            col_nn = f[nc_column].values
            col_v, col_j = f[v_beta_column].values, f[j_beta_column].values
            col_v,col_j = [ v if isinstance(v, float) and math.isnan(v) else v[0] + v[2:5] + v[6:] if v[5] == '0' else v[0] + v[2:] for v in col_v], [j if isinstance(j, float) and math.isnan(j) else j[0] + j[2:5] + j[6:] if j[5] == '0' else j[0] + j[2:]  for j in col_j]
            col_v,col_j = [v if isinstance(v, float) and math.isnan(v) else v[:-2] + v[-1] if v[-2] == '0' else v for v in col_v], [j if isinstance(j, float) and math.isnan(j) else j[:-2] +j[-1] if j[-2] == '0' else j for j in col_j]
            col_beta = f[aa_column_beta].values if in_frame else f[nc_column]
            frame_bool = f[frame_column].values 
            col_v_al,col_j_al = f[v_allele_column].values,f[j_allele_column].values
            col_v_del, col_j_del = f[v_deletion_column].values, f[j_deletion_column].values
            col_id = f[patient_id_column].values
       
            
        for k in range(len(col_beta)):
      
            if math.isnan(col_v_al[k]) or math.isnan(col_j_al[k]):
                continue
            if not PassFiltered_((col_beta[k],col_v[k],col_j[k]),frame_bool[k],in_frame,max_length,filter_genes):
                continue  
            if col_nn[k] in unique_seqs:
                continue
            beta_sequences.append(col_beta[k])
            unique_seqs.add(col_nn[k])
            v_beta.append(col_v[k])
            j_beta.append(col_j[k])
            nn_beta.append(col_nn[k])
            v_al_beta.append(col_v_al[k])
            j_al_beta.append(col_j_al[k])
            v_deletion.append(col_v_del[k])
            j_deletion.append(col_j_del[k])
            patient_id.append(col_id[k])
        print("The Number of the sequences: {}".format(len(beta_sequences)))

        res = pd.DataFrame(columns=['patient_id','seq','nn','v', 'v_allele', 'v_deletions', 'j', 'j_allele', 'j_deletions'])
        res['seq'] = beta_sequences
        res['v'] = v_beta
        res['j'] = j_beta
        res['nn'] = nn_beta
        res['v_allele'] = v_al_beta
        res['j_allele'] = j_al_beta
        res['patient_id'] = patient_id
        res['v_deletions'] = v_deletion
        res['j_deletions'] = j_deletion
        res.to_csv('data/whole_seqs_nn.tsv',sep='\t',mode = 'a',index=False,header=not os.path.isfile('data/whole_seqs_nn.tsv'))
    

    f = pd.read_csv('data/whole_seqs_nn.tsv',sep=seps[i])
    train_set = f.sample(frac= train_frac, random_state = np.random.RandomState(seed=42))
    test_set = f.drop(train_set.index)

    train_set.to_csv('data/whole_seqs_nn_train.tsv', index= False, sep='\t')
    test_set.to_csv('data/whole_seqs_nn_test.tsv', index=False, sep='\t')
       

def PassFiltered_(to_filter, frame_bool,in_frame,max_length,filter_genes):
    ### to check whether the row of data is valid
    seq,v,j = to_filter
    if in_frame:
        if frame_bool != 'In':
            return False             
    if type(seq) is not str:            
        return False
    if type(v) is not str or type(j) is not str:
        return False
    if len(seq) > max_length or '*' in seq or 'C' != seq[0] or ('F' != seq[-1] and 'YV' != seq[-2:]):
        return False
    if v =='unresolved' or j == 'unresolved':
        return False

    if j == 'TCRBJ02-05' : #based on data preprocessing step in soNNia
        return False
    if seq =='CFFKQKTAYEQYF': #based on data preprocessing step in soNNia
        return False
    if filter_genes:
        if (v not in vs_default) or (j not in js_default):
            return False
    return True

if __name__ == '__main__':
    get_whole_data(['original_data/' + di for di in os.listdir('original_data')])
    