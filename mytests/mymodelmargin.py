from transformers import PreTrainedModel, GPT2Model, GPT2LMHeadModel, GPT2Config
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch 
import numpy as np 
import torch
from mymodelnew4 import MyModel

class MySimilarityModel(PreTrainedModel):
    config_class = GPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.final_embed_size = 512
        #self.config= config
        #self.encoder = GPT2Model(config).to('cuda')
        print('Im here!')
        self.encoder = MyModel.from_pretrained('./nodotmodel3_weights_2024-08-10--20:06:46alaki').encoder
        print('Im after that!')
        #self.encoder.eval()
        self.linear_embedding = torch.nn.Linear(config.hidden_size, self.final_embed_size, bias=False).to('cuda')
        self.cos = torch.nn.CosineSimilarity(dim=1)
    def forward(self, list_of_sub_batches, sign, indices):
        sub_batch_count = len(list_of_sub_batches)
        all_hidden_embeddings = torch.tensor([]).to('cuda')
        for i in range(sub_batch_count):
            pivot_twodim = ((list_of_sub_batches[i] == self.config.eos_token_id).cumsum(1) == 1)
            pivot_index = pivot_twodim.nonzero()[:,1]
            #with torch.no_grad():
            encoder_outputs = self.encoder(list_of_sub_batches[i])
            hidden_embedding = encoder_outputs.last_hidden_state[np.arange(encoder_outputs.last_hidden_state.shape[0]),pivot_index,:]
            all_hidden_embeddings = torch.cat((all_hidden_embeddings, hidden_embedding), dim=0)
        #print('len indices is ', len(indices))
        loss = 0
        #print('hello', np.array(sign) == 'base')
        similar_vectors = []
        random_vectors = []
        for count in range(max(indices) + 1):
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
            similars_vector = self.cos(self.linear_embedding(similar_embeddings),
                                                                   self.linear_embedding(base_embedding).expand(similar_embeddings.shape[0], -1))
            min_of_similars = 0
            max_of_randoms = 0
            if similars_vector.numel() > 0:
                min_of_similars = torch.min(similars_vector)
            randoms_vector = self.cos(self.linear_embedding(random_embeddings), 
                                                   self.linear_embedding(base_embedding).expand(random_embeddings.shape[0], -1))
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
       