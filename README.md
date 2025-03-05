The project was done during a research internship in the Computational Systems Biology research group at the Department of Computer Science at Aalto University.

# TCR Transformer

<!--project can be found on the location ```/scratch/cs/csb/users/huttuna6/TCR_transformer```on triton. There you can also found more files like trained models.-->

Deep autoregressive model predicting CDR3 amino acid sequences based on V and J genes amino acid sequences.
The model is based on the BERT model i.e. transformer models encoder part. 

The base for the model is pre-trained [protBERT model](https://huggingface.co/Rostlab/prot_bert_bfd) trained by Rostlab. The weights and bias of the protBERT model are copied to Pytorch implementation found in ```code/CustomprotBERT.py```. After copying the weights and bias the model is finetuned for the CDR3 prediction task. 

## The structure of the repo
```code/```
   
   Python files are used to process Emerson data, process data before training, train the model, generate CDR3 sequences, and evaluate the model. Outputs of run scripts are saved in the logs folder created when running a batch script.

```data/```

  Contains files needed to process the data. Processed Emerson data will be saved here.

```scripts/```

  Jupyter notebooks were utilised in the developing process. REMOVE THESE before publishing the repo.

## Setup the project 

All needed Python packages can be installed using conda environment.

1. On triton load miniconda: ```module load miniconda```
2. Create environment using ```environment.yml``` file: ```conda create env -f environment.yml```
3. After the environment is solved, it can be activated with the command ```source activate TCR-env``` and the environment can be deactivated with the command ```conda deactivate```

## Training the model 

The model can be trained by running script ```code/run_train.sh``` on triton. Remember to add your email to the script to get an email when training is started or fails. You can also modify other slurm parameters. 

### Steps of the training 
1. As an input, the model takes tokenized TCR sequence $[V_1, ..., V_v] [CDR_1, CDR_2, ..., CDR_N] [J_1, ..., J_j]$ 
2. Tokenized sequences are embedded and positionally encoded $\rightarrow$ tensor
3. The tensor is inputted to the transformer encoder block
4. The resulting representations of $[V_v, CDR_1, ..., CDR_{N-1}]$ are extracted
5. The final CDR3 sequences are predicted based on that


