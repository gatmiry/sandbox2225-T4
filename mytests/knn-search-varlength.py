from datasets import load_from_disk
import numpy as np
import torch
#from transformers import AutoTokenizer

num_train_text = 30000
num_test_text = 2000
device = 'cuda'
#tokenizer = AutoTokenizer.from_pretrained('gpt2')
text_datasets = load_from_disk('./tinystories_dataset')
text_datasets = {
    'train': text_datasets['train'].select(np.arange(num_train_text)),#500000
    'validation': text_datasets['validation'].select(np.arange(num_test_text))#5000
}

from calculate_embedding import EmbedCalculator
calculator = EmbedCalculator()
batch_size = 20
num_batch = int(num_test_text / batch_size)
all_hidden_embeddings = torch.empty(0, calculator.hidden_size).to(device)
#all_hidden_embeddings = torch.empty(0, 3072).to(device)
for i in range(num_batch):
    #print('len is ', text_datasets['validation']['text'][i*batch_size:(i+1)*batch_size])
    hidden_embeddings = calculator.calculate_embedding(text_datasets['validation']['text'][i*batch_size:(i+1)*batch_size]) 
    all_hidden_embeddings = torch.cat((all_hidden_embeddings, hidden_embeddings), dim=0)

np_hidden_embeddings = np.array(all_hidden_embeddings.to('cpu')).astype('float32')
np_hidden_embeddings.shape

### time to find the nearest neighbor
base_index = 3
input_text = text_datasets['validation']['text'][base_index]
print('INPUT TEXT IS: ', input_text)
import faiss
index_flat = faiss.IndexFlatL2(calculator.hidden_size)
res = faiss.StandardGpuResources()
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
gpu_index_flat.add(np_hidden_embeddings)
k = 2
D, I = gpu_index_flat.search(np.expand_dims(np_hidden_embeddings[base_index], axis=0), k)
close_text = text_datasets['validation']['text'][I[0][1]]
print('CLOSE TEXT INDEX IS: ' + str(I[0][1]))
print('CLOSE TEXT IS: ', close_text)