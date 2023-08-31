import torch
import numpy as np

class CDR3Dataset(torch.utils.data.Dataset):
    def __init__(self, V, CDR3, J, tgt, tokenizer, evaluate=False):

        self.tokenizer = tokenizer
        self.V = V
        self.CDR3 = CDR3
        self.J = J
        self.labels = tgt
        self.evaluate = evaluate
        
        if len(tgt) == 1:
            self.max_length = len(tgt[0].split()) + 3
            self.CDR3_max_length = len(CDR3[0].split())
            # self.V_max_length = len(V[0].split())
            # self.J_max_length = len(J[0].split())
        else:
            length = 0
            V_length = 0
            CDR3_length = 0
            J_length = 0
            for i in range(len(tgt)):
                length = max(length, len(tgt[i].split())+3)
                CDR3_length = max(CDR3_length, len(CDR3[i].split()))
                # V_length = max(V_length, len(V[i].split()))
                # J_length = max(J_length, len(J[i].split()))

            self.max_length = length
            self.CDR3_max_length = CDR3_length
           # self.V_max_length = V_length +1 #[CLS]
           # self.J_max_length =  J_length + 2 #[SEP] and one [PAD]
           # print(self.CDR3_max_length)
 
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.labels[idx], padding= 'max_length', max_length= self.max_length, return_tensors='pt', return_attention_mask=True)
        padded_tokenized_seq = tokenized['input_ids']
        pad_mask = ~tokenized['attention_mask'].to(torch.bool)
        
        
        v_len = len(self.V[idx].replace(" ", "")) + 1 # Tokenizer adds [CLS] token at the beginning of the sequence
        cdr_len = len(self.CDR3[idx].replace(" ", ""))
        j_len = padded_tokenized_seq.shape[-1]- v_len- cdr_len
        lengths = torch.tensor([v_len, cdr_len])
        
        
        if self.evaluate:
            CDR3_label = torch.cat((padded_tokenized_seq[:, v_len: v_len+cdr_len], torch.full((1, self.CDR3_max_length - len(self.CDR3[idx].split())), 0)), dim=-1)
            item = {'seq': padded_tokenized_seq.squeeze(0),
                    'length': lengths,
                    #'mask': None,
                    'pad_mask': pad_mask.squeeze(0),
                    'CDR3_label': CDR3_label.squeeze(0)}
            
        else:
            CDR3_label = torch.cat((padded_tokenized_seq[:, v_len: v_len+cdr_len], torch.full((1, self.CDR3_max_length - len(self.CDR3[idx].split())), -100)), dim=-1)
            mask = self.subsequent_mask(v_len, cdr_len, j_len)
        
            item = {'seq': padded_tokenized_seq.squeeze(0),
                    'length': lengths,
                    'mask': mask,
                    'pad_mask': pad_mask.squeeze(0),
                    'CDR3_label': CDR3_label.squeeze(0)}
        
        return item
    
    def subsequent_mask(self, v, cdr3, j):

        cdr3_len = cdr3
        v_len = v
        j_len = j

        v_block1 = np.zeros(shape=(v_len, v_len))
        v_block2 = np.zeros(shape=(cdr3_len, v_len))
        v_block3 = np.zeros(shape=(j_len, v_len))

        j_block1 = np.zeros(shape=(v_len, j_len))
        j_block2 = np.zeros(shape=(cdr3_len, j_len))
        j_block3 = np.zeros(shape=(j_len, j_len))

        cdr3_block1 = np.full(shape=(v_len, cdr3_len), fill_value=np.NINF)
        cdr3_block2 = np.triu(np.full((cdr3_len, cdr3_len), np.NINF), k = 0)
        cdr3_block3 = np.full(shape=(j_len, cdr3_len), fill_value=np.NINF)

        row1 = np.concatenate((v_block1, cdr3_block1, j_block1), axis=1)
        row2 = np.concatenate((v_block2, cdr3_block2, j_block2), axis=1)
        row3 = np.concatenate((v_block3, cdr3_block3, j_block3), axis=1)

        attention_mask = np.concatenate((row1, row2, row3), axis =0)

        return torch.from_numpy(attention_mask) != 0