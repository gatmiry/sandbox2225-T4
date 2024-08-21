from transformers import PreTrainedModel, GPT2Model, GPT2LMHeadModel, GPT2Config
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch 
import numpy as np 

class MyModel(PreTrainedModel):
    config_class = GPT2Config

    def __init__(self, config):
        super().__init__(config)
        #self.config= config
        self.encoder = GPT2Model(config)
        #self.second_encoder = GPT2Model(config)
        self.decoder = GPT2Model(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False).to('cuda')
        #self.max_num_paragraphs = max_num_paragraphs
        
        ### these are for the embedding weights
        #self.embedding_wte = torch.nn.Embedding(config.vocab_size, config.n_embd)
        #elf.embedding_wpe = torch.nn.Embedding(config.max_position_embeddings, config.n_embd)
    
    def set_num_embeddings(self, num_embeddings):
        self.num_embeddings = num_embeddings
        for i, block in enumerate(self.h):
            block.set_num_embeddings(num_embeddings)

    def forward(self, input_ids, labels=None, attention_mask=None):
        hidden_embedding = 0
        total_loss = 0
        logits_pack = []
        num_paragraphs = len(input_ids)
        for i in range(1, num_paragraphs):
            
            pivot_twodim = ((input_ids[i] == self.config.eos_token_id).cumsum(1) == 1)
            pivot_index = pivot_twodim.nonzero()[:,1]
       # print('input ids masked is ',input_ids[np.arange(input_ids.shape[0]),pivot_index-2])

            position_ids = torch.arange(input_ids[i].shape[1], dtype=torch.long, device='cuda')
            position_ids = position_ids.unsqueeze(0)
            inputs_embeds = self.decoder.wte(input_ids[i])
            position_embeds = self.decoder.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds

            encoder_outputs = self.encoder(input_ids[i-1])
        #print('first len is ',encoder_outputs.last_hidden_state.shape[0], ' second len is ', len(pivot_index))
            hidden_embedding += encoder_outputs.last_hidden_state[np.arange(encoder_outputs.last_hidden_state.shape[0]),pivot_index,:].unsqueeze(1)
            updated_input = torch.cat((hidden_embedding, hidden_states), dim=1)
            final_hidden_states = self.decoder(inputs_embeds=updated_input)[0]
            logits = self.lm_head(final_hidden_states)
        #logits = F.log_softmax(logits, dim=-1)
            shifted_prediction_scores = logits[:, 1:-1, :]
        
            if labels == None:
                return {'logits':logits}

            labels[i][attention_mask[i] == 0] = -100
            labels[i][pivot_twodim] = self.config.eos_token_id
        #print('labels are ', labels[0][])
            labels[i] = labels[i][:, 1:]
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.contiguous().view(-1, self.config.vocab_size), labels[i].contiguous().view(-1))
            total_loss += lm_loss
            logits_pack.append(logits[:,1:,:])
        return {'loss': lm_loss, 'logits_pack':logits_pack}