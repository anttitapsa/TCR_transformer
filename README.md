# TCR_transformer

Deep autoregressive model predicting CDR3 amino acid sequences based on V and J genes amino acid sequences.
The model is based on the BERT model i.e. transfromer models encoder part. 

The base for the model is pre-trained [protBERT model](https://huggingface.co/Rostlab/prot_bert_bfd) trained by Rostlab. The weights and bias of protBERT model are copied to Pytorch implementation found in ```code/CustomprotBERT.py```. After copying the weights and bias the model is finetuned for CDR3 prediction task. 

## Setup the project 

All needed Python packages can be installed  

## Training the model 

Model can be trained by running script ```code/run_train.sh``` on triton. Remember to add your email to script to get email when training is started or failed. You can also modify other slurm parameters. 

### Steps of the training 
1. As an input, model take tokenized TCR sequence $[V_1, ..., V_v] [CDR_1, CDR_2, ..., CDR_N] [J_1, ..., J_j]$ 
2. Tokenized sequences is embedded and positionally encoded $\rightarrow$ tensor
3. Tensor is inputted to transformer encoder block
4. The resulting representations of $[V_v, CDR_1, ..., CDR_{N-1}]$ are extracted
5. The final CDR3 sequences are predicted based on that

### How to Install and Run the Project
Test commit. New test commit.

### How to Use the Project
