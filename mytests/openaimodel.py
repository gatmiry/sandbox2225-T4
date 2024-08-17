from transformers import AutoTokenizer
import torch 
import numpy as np 
import torch
import requests 

class OpenAIModel:

    def __init__(self):
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.header = {'Authorization': '0NNFKdThRKtvSrCg5ApslcsVeg97r3bj'}
        self.data = {'model': 'text-embedding-3-large', 'input': ['xxx', 'asdfe']}
        self.url = 'https://blog.yintat.com/dqjoBCx0P2k7/api/models/unified-completion'
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')

    def __call__(self, list_of_sub_batches, sign, indices):
        sub_batch_count = len(list_of_sub_batches)
        all_hidden_embeddings = torch.tensor([])
        for i in range(sub_batch_count):
            sub_batch = list_of_sub_batches[i]
            for j in range(sub_batch.shape[0]):
                print('im here i is ', i, ' j is ', j)
                self.data['input'] = [self.tokenizer.decode(sub_batch[j], skip_special_tokens=True)]
                response = requests.post(self.url, headers=self.header, json=self.data, timeout=600)
                import json
                data = json.loads(response.text)
                hidden_embedding = data['data'][0]['embedding']
                all_hidden_embeddings = torch.cat((all_hidden_embeddings, torch.tensor(hidden_embedding).unsqueeze(0)), dim=0)
        #print('len indices is ', len(indices))
        loss = 0
        #print('hello', np.array(sign) == 'base')
        similar_vectors = []
        random_vectors = []
        for count in range(max(indices) + 1):
            print('im in the seocnd loop count is '. count)
            base_embedding = all_hidden_embeddings[(np.array(sign) == 'base') & (np.array(indices) == count), :]
            similar_embeddings = all_hidden_embeddings[(np.array(sign) == 'similar') & (np.array(indices) == count), :]
            #print('similar_embedding shape is ', similar_embeddings.shape)
            #print('the other one is ', np.array(indices) == count, ' count is ', count, ' indices is ', indices)
            
            random_embeddings = all_hidden_embeddings[(np.array(sign) == 'random') & (np.array(indices) == count), :]
            #print('avali is ', self.linear_embedding(similar_embeddings).shape)
            #print('dovomi is ', self.linear_embedding(base_embedding).expand(similar_embeddings.shape[0], -1).shape)
            #print('min arg is ', self.cos(self.linear_embedding(similar_embeddings),
                                                                   #self.linear_embedding(base_embedding).expand(similar_embeddings.shape[0], -1)))
            #print('min arg shape is ', self.cos(self.linear_embedding(similar_embeddings),
                                                                   #self.linear_embedding(base_embedding).expand(similar_embeddings.shape[0], -1)).shape)
            similars_vector = self.cos(similar_embeddings, base_embedding.expand(similar_embeddings.shape[0], -1))
            min_of_similars = 0
            max_of_randoms = 0
            if similars_vector.numel() > 0:
                min_of_similars = torch.min(similars_vector)
            randoms_vector = self.cos(random_embeddings, base_embedding.expand(random_embeddings.shape[0], -1))
            if randoms_vector.numel() > 0:
                max_of_randoms = torch.max(randoms_vector)
            max_margin = max_of_randoms - min_of_similars
            #print('max margin is ', max_margin)
            loss += max_margin
            similar_vectors.append(similars_vector)
            random_vectors.append(randoms_vector)
            ## calculating the top k 
            #all_vector = torch.cat((similars_vector, randoms_vector), dim=0)
            #_, top_k_indices = torch.topk(all_vector, self.k)
        return {'loss': loss, 'similar_scores': similar_vectors, 'random_scores': random_vectors}
       