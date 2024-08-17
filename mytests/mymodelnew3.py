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
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False).to('cuda')

    def forward(self, input_ids, labels=None, attention_mask=None):
        #device = 'cuda'
        ## find the first eos index
        #print('example input is ', len(input_ids[0]),' ', len(input_ids[1]))
        #print('eos token id is ', self.config.eos_token_id)
        pivot_index = []
        for i in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                if input_ids[i][j] == self.config.eos_token_id:
                    #print('i is ', i, ' j is ', j)
                    pivot_index.append(j)
                    break
            #pivot_index.append(j)

        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device='cuda')
        position_ids = position_ids.unsqueeze(0)
        inputs_embeds = self.decoder.wte(input_ids)
        position_embeds = self.decoder.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        encoder_outputs = self.encoder(input_ids)
        #print('first len is ',encoder_outputs.last_hidden_state.shape[0], ' second len is ', len(pivot_index))
        hidden_embedding = encoder_outputs.last_hidden_state[np.arange(encoder_outputs.last_hidden_state.shape[0]),pivot_index,:].unsqueeze(1)
        # just to obtain the hidden embeddings
        #with torch.no_grad():
        #    decoder_hidden_inputs = self.second_encoder(input_ids, output_hidden_states=True).hidden_states[0]
        #hidden_embedding_dim = hidden_embedding.shape[2]
        updated_input = torch.cat((hidden_embedding, hidden_states), dim=1)
        final_hidden_states = self.decoder(inputs_embeds=updated_input)[0]
        logits = self.lm_head(final_hidden_states)
        #logits = F.log_softmax(logits, dim=-1)
        shifted_prediction_scores = logits[:, 1:-1, :]
        print('attention mask is ', attention_mask)
        if labels == None:
            return {'logits':logits}

        labels[attention_mask == 0] = -100 
        labels = labels[:, 1:]
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(shifted_prediction_scores.contiguous().view(-1, self.config.vocab_size), labels.contiguous().view(-1))
        return {'loss': lm_loss, 'logits':logits[:,1:,:]}