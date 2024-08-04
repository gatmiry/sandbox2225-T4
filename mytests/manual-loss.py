## prepare the model and tokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
from datasets import load_dataset
tinystories =  load_dataset("roneneldan/TinyStories")
tinystories
from mymodelnew3 import MyModel
checkpoint = './posmodel3_weights_2024-08-01--05:36:40alaki'
model2 = MyModel.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

## prepare the dataset
from datasets import load_from_disk
import numpy as np
tokenized_datasets = load_from_disk('./model3-outputs-gpt2tokenizer-varlength')
## slicing the dataset to a smaller size
tokenized_datasets = {
    'train': tokenized_datasets['train'].select(np.arange(30000)),#500000
    'validation': tokenized_datasets['validation'].select(np.arange(20000))#5000
}

ave = 0
for i in range(0,10):
    #input_str = tokenized_datasets['train']['input_ids'][i]
    #input_ids = tokenizer(input_str, return_tensors='pt')['input_ids']
    input_ids = torch.tensor(tokenized_datasets['validation']['input_ids'][3500 + i*1:3500 + (i+1)*1]).to('cuda')
    print('input id shape is ', input_ids.shape)
    model2 = model2.to('cuda')
    model2.eval()
    with torch.no_grad():
        print('im here!')
        loss_val = model2(input_ids, labels=input_ids)['loss']
        ave += loss_val
        print('i is ', i, ' loss is ', loss_val)
ave = ave / 10
print('average loss is ', ave)