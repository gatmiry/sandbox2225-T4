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

tokenizer.pad_token = tokenizer.eos_token
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
#myten = data_collator(tokenized_datasets['validation']['input_ids'][3500 + 1*20:3500 + 2*20], return_tensors='pt')
#print('myten is ', myten.shape)

ave = 0
for i in range(0,10):
    #input_str = tokenized_datasets['train']['input_ids'][i]
    #input_ids = tokenizer(input_str, return_tensors='pt')['input_ids']
    input_ids = data_collator(tokenized_datasets['validation']['input_ids'][3500 + i*20:3500 + (i+1)*20])['input_ids'].to('cuda')
    print('input id shape is ', input_ids.shape)
    model2 = model2.to('cuda')
    model2.eval()
    with torch.no_grad():
        print('im here!')
        mymask = torch.where(input_ids == tokenizer.eos_token_id, 0, 1)
        print('mask is ', mymask)
        loss_val = model2(input_ids, labels=input_ids, attention_mask=mymask)['loss']
        ave += loss_val
        print('i is ', i, ' loss is ', loss_val)
ave = ave / 10
print('average loss is ', ave)