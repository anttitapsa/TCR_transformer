import pickle
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertModel, AutoTokenizer


from process_data import process_data
from CDR3Dataset import CDR3Dataset
from CustomProtBert import CustomprotBERT

def load_model(save_dir, device, lr):
    checkpoint = torch.load(f'{save_dir}_checkpoint.pth', map_location= device)
    model = checkpoint['model']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_batch =  checkpoint['batch']
    start = checkpoint['epoch']
    return model, optimizer, start_batch, start

def train(train_set,
          save_dir,
          batch_size = 64,
          EPOCHS = 10,
          lr = 1e-6,
          device='cpu',
          continue_train = False):

    
    train_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size)

    start = 0
    start_batch = 0

    EPOCHS = EPOCHS
    criterion = torch.nn.CrossEntropyLoss()

    if continue_train:
        print('Loading model...')
        model, optimizer, start_batch, start = load_model(save_dir, device, lr)
        loss_df = pd.read_csv(f'{save_dir}_loss_PPL_train.tsv', sep='\t')
        print(f'Continue training from epoch {start+1}, batch {start_batch+1}')
    else:
        model = CustomprotBERT(base_model, train_set.CDR3_max_length)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_df = pd.DataFrame(columns=["epochs","batch", "loss/train", "PPL/train"])
        print('Starting training...')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    model = model.to(device)


    for epoch in range(start, EPOCHS):
        for idx, batch in enumerate(train_loader):

            if epoch == start and idx < start_batch:
                continue
    
            batch = {k: v.to(device) for k, v in batch.items()}
            
            logits = model(src = batch['seq'],
                       length = batch['length'],
                       pad_mask = batch['pad_mask'],
                       mask = batch['mask'],
                       device = device)

        
            loss = criterion(logits.transpose(1,2), batch['CDR3_label'])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            perplexity = torch.exp(loss)
            
            loss_df = pd.concat([loss_df, pd.DataFrame({"epochs": [epoch] ,"batch": [idx], "loss/train": [loss.to('cpu').detach().numpy()], "PPL/train":[perplexity.to('cpu').detach().numpy()]})])
            loss_df.to_csv(f'{save_dir}_loss_PPL_train.tsv', sep='\t', index=False)

            if idx%100 == 0:
                print(f'{epoch+1}/{EPOCHS} epoch, batch: {idx +1}, loss: {loss.item()}, PPL: {perplexity}')
                torch.save({
                'epoch': epoch,
                'batch' : idx,
                'model': model.module,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f"{save_dir}_checkpoint.pth")
            
    torch.save({'model': model.module,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f"{save_dir}.pth")
        

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    '''
    # Argument parser to be finished
    parser = argparse.ArgumentParser()
    parser.add_argument("-lm", "--loadmodel", action = "store_true")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-lr", "--learningrate", type=float, default=0.002)
    parser.add_argument("-n", "--name", required=True, default="model")
    parser.add_argument("remainder", nargs="*")
    args = parser.parse_args()
    '''

    model_name = "030823_customprotBERT_parallel"
    info_df = pd.DataFrame(columns=["epochs","batch", "loss/train", "PPL/train"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_model = BertModel.from_pretrained('Rostlab/prot_bert_bfd')
    tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd')
    #config = BertConfig.from_pretrained('Rostlab/prot_bert_bfd')

    '''
    #use this block insted of line 129 if you have saved processed data
    with open('../../scripts/train_set_.pkl', 'rb') as f:
        V_data, CDR3_data, J_data,tgt_data = pickle.load(f)
    f.close()
    '''
    V_data, CDR3_data, J_data,tgt_data = process_data()
    print("Data collected")

    train_set = CDR3Dataset(V_data, CDR3_data, J_data, tgt_data, tokenizer)

    #model = CustomprotBERT(base_model, train_set.CDR3_max_length)

    train(train_set = train_set,
          save_dir = model_name,
          batch_size = 320,
          EPOCHS = 10,
          lr = 1e-6,
          device=device,
          continue_train = True)