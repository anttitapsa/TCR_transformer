import transformers
from transformers import BertConfig 
import torch.nn as nn
import torch

import transformers 
import torch.nn as nn

class CustomprotBERT(transformers.PreTrainedModel):
    def __init__(self, protBERT, CDR3_max_length):
        super(CustomprotBERT, self).__init__(config=BertConfig.from_pretrained('Rostlab/prot_bert_bfd'))
        
        self.config = protBERT.config
        self.CDR3_max_length = CDR3_max_length
        self.embeddings = protBERT.embeddings
        self.encoder= nn.TransformerEncoder(encoder_layer= nn.TransformerEncoderLayer(  d_model=self.config.hidden_size,
                                                                                        nhead=self.config.num_attention_heads,
                                                                                        dim_feedforward=self.config.intermediate_size,
                                                                                        dropout=self.config.hidden_dropout_prob,
                                                                                        activation=self.config.hidden_act,
                                                                                        layer_norm_eps=self.config.layer_norm_eps,
                                                                                        batch_first=True),
                                          num_layers = self.config.num_hidden_layers)
        self.fc = nn.Linear(self.config.hidden_size, self.config.vocab_size)

        self.initialize_weights(protBERT)

    def forward(self, src, length, pad_mask=None,  mask=None, device = 'cpu'):
        #print(f'input shape:{src.shape}')
        embed = self.embeddings(src)
        mask = mask.repeat((self.config.num_attention_heads,1,1)) if mask != None else mask
        hidden_state = self.encoder(src=embed,
                                    mask=mask,
                                    src_key_padding_mask =pad_mask)
        
        #v_idx = [i.shape[-1] -1 for i in v] # Indeces of the last value of the v gene
        #cdr3_idx = [i + self.CDR3_max_length for i in v_idx] # Indeces of the last residue of the CDR3
        
        cdr3_hidden_state = torch.zeros((hidden_state.shape[0], self.CDR3_max_length, hidden_state.shape[-1])).to(device)
        
        for i in range(hidden_state.shape[0]):
            cdr3_len = length[i,1]
            pad_len = self.CDR3_max_length -cdr3_len
            cdr3 = hidden_state[i, length[i,0]-1:length[i,0]-1+ length[i,1],:]
            pad_seq = torch.full((pad_len, hidden_state.shape[-1]), 0).to(device)
            cdr3 = torch.cat((cdr3, pad_seq), axis=0)
            cdr3_hidden_state[i,:,:] = cdr3 #1024 dim presentation of [V_v, CDR_1, ..., CDR_{N-1}]
        
        cdr3_logits = self.fc(cdr3_hidden_state)
        
        return cdr3_logits   

    
    def initialize_weights(self, protBERT):

            #this implementation works so that theresulting tensor is same as using 
            #Huggingface's model on the fourth decimal. Tensors are same in fifth decimal over 99.99%

            protBERT_encoder_weights = protBERT.encoder.state_dict()
            with torch.no_grad():
                for i in range(30):
                    weights = {
                        'att_weight': torch.cat((protBERT_encoder_weights[f'layer.{i}.attention.self.query.weight'],
                                                    protBERT_encoder_weights[f'layer.{i}.attention.self.key.weight'],
                                                    protBERT_encoder_weights[f'layer.{i}.attention.self.value.weight']),
                                                    axis=0),
                        'att_bias': torch.cat((protBERT_encoder_weights[f'layer.{i}.attention.self.query.bias'],
                                                    protBERT_encoder_weights[f'layer.{i}.attention.self.key.bias'],
                                                    protBERT_encoder_weights[f'layer.{i}.attention.self.value.bias']),
                                                    axis=0),
                        'att_out_weight': protBERT_encoder_weights[f'layer.{i}.attention.output.dense.weight'],
                        'att_out_bias': protBERT_encoder_weights[f'layer.{i}.attention.output.dense.bias'],
                        'interm_weight': protBERT_encoder_weights[f'layer.{i}.intermediate.dense.weight'],
                        'interm_bias': protBERT_encoder_weights[f'layer.{i}.intermediate.dense.bias'],
                        'out_weight': protBERT_encoder_weights[f'layer.{i}.output.dense.weight'],
                        'out_bias': protBERT_encoder_weights[f'layer.{i}.output.dense.bias'],
                        'norm1_weight': protBERT_encoder_weights[f'layer.{i}.attention.output.LayerNorm.weight'],
                        'norm1_bias': protBERT_encoder_weights[f'layer.{i}.attention.output.LayerNorm.bias'],
                        'norm2_weight': protBERT_encoder_weights[f'layer.{i}.output.LayerNorm.weight'],
                        'norm2_bias': protBERT_encoder_weights[f'layer.{i}.output.LayerNorm.bias'],
                    }

                    #print(weights['out_weight'].dtype)
                    #print(self.encoder.layers[i].self_attn.in_proj_weight.dtype)

                    self.encoder.layers[i].self_attn.in_proj_weight.copy_(weights['att_weight'])
                    self.encoder.layers[i].self_attn.in_proj_bias.copy_(weights['att_bias'])
                    self.encoder.layers[i].self_attn.out_proj.weight.copy_(weights['att_out_weight'])
                    self.encoder.layers[i].self_attn.out_proj.bias.copy_(weights['att_out_bias'])
                    self.encoder.layers[i].linear1.weight.copy_(weights['interm_weight'])
                    self.encoder.layers[i].linear1.bias.copy_(weights['interm_bias'])
                    self.encoder.layers[i].linear2.weight.copy_(weights['out_weight'])
                    self.encoder.layers[i].linear2.bias.copy_(weights['out_bias'])
                    self.encoder.layers[i].norm1.weight.copy_(weights['norm1_weight'])
                    self.encoder.layers[i].norm1.bias.copy_(weights['norm1_bias'])
                    self.encoder.layers[i].norm2.weight.copy_(weights['norm2_weight'])
                    self.encoder.layers[i].norm2.bias.copy_(weights['norm2_bias'])