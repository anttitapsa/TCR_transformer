# TCR_transformer

project can be found on the location ```/scratch/cs/csb/users/huttuna6/TCR_transformer```on triton. There you can also found more files like trained models.

Deep autoregressive model predicting CDR3 amino acid sequences based on V and J genes amino acid sequences.
The model is based on the BERT model i.e. transfromer models encoder part. 

The base for the model is pre-trained [protBERT model](https://huggingface.co/Rostlab/prot_bert_bfd) trained by Rostlab. The weights and bias of protBERT model are copied to Pytorch implementation found in ```code/CustomprotBERT.py```. After copying the weights and bias the model is finetuned for CDR3 prediction task. 

## The structure of the repo
```code/```
   
   Python files to for prosessing Emerson data, process data before training, train model, generate CDR3 sequences, and evaluate model.

```data/```

  Contains files needed to processing the data. Processed Emerson data will be saved here.

```scripts/```

  Jupyter notebooks utilised as in the developing process. REMOVE THESE before publishing the repo.

## Setup the project 

All needed Python packages can be installed using conda environment.

1. On triton load miniconda: ```module load miniconda```
2. Create environment using ```environment.yml``` file: ```conda create env -f environment.yml```
3. After the enviroment is solved, it can be activated with command ```source activate TCR-env``` and the environment can be deactivated with command ```conda deactivate```

## Training the model 

Model can be trained by running script ```code/run_train.sh``` on triton. Remember to add your email to script to get email when training is started or failed. You can also modify other slurm parameters. 

### Steps of the training 
1. As an input, model take tokenized TCR sequence $[V_1, ..., V_v] [CDR_1, CDR_2, ..., CDR_N] [J_1, ..., J_j]$ 
2. Tokenized sequences is embedded and positionally encoded $\rightarrow$ tensor
3. Tensor is inputted to transformer encoder block
4. The resulting representations of $[V_v, CDR_1, ..., CDR_{N-1}]$ are extracted
5. The final CDR3 sequences are predicted based on that


