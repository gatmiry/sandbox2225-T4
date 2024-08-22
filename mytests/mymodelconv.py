from transformers import PreTrainedModel, GPT2Model, GPT2LMHeadModel, GPT2Config, AutoTokenizer
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
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        #self.max_num_paragraphs = max_num_paragraphs
        
        ### these are for the embedding weights
        #self.embedding_wte = torch.nn.Embedding(config.vocab_size, config.n_embd)
        #elf.embedding_wpe = torch.nn.Embedding(config.max_position_embeddings, config.n_embd)
    
    def set_num_embeddings(self, num_embeddings):
        self.num_embeddings = num_embeddings
        for i, block in enumerate(self.encoder.h):
            block.attn.set_num_embeddings(num_embeddings)
        for i, block in enumerate(self.decoder.h):
            block.attn.set_num_embeddings(num_embeddings)    

    def forward(self, input_ids_list, labels_list=None, attention_mask_list=None):
        hidden_embedding = 0
        total_loss = 0
        logits_pack = []
        num_paragraphs = len(input_ids_list)
        print('num paragraphs are ', num_paragraphs)
        #print('len input id is ', len(input_ids_list))
        for i in range(1, num_paragraphs):
            #print('im here!!!')
            #print('input_ids[i-1] is ', input_ids_list[i-1].shape)
            pivot_twodim = ((input_ids_list[i-1] == self.config.eos_token_id).cumsum(1) == 1)
            torch.set_printoptions(profile="full")
            #print('pivot_twodim is ', input_ids_list[i-1])
            torch.set_printoptions(profile="default")
            #print('pivot index is ', pivot_twodim.nonzero())
            #print('decoded input is ', self.tokenizer.decode(input_ids_list[i-1][0]))
            pivot_index = pivot_twodim.nonzero()[:,1]
       # print('input ids masked is ',input_ids[np.arange(input_ids.shape[0]),pivot_index-2])

            position_ids = torch.arange(input_ids_list[i].shape[1], dtype=torch.long, device='cuda')
            position_ids = position_ids.unsqueeze(0)
            inputs_embeds = self.decoder.wte(input_ids_list[i])
            position_embeds = self.decoder.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds

            encoder_outputs = self.encoder(input_ids_list[i-1])
            #print('this is ', encoder_outputs.last_hidden_state.shape, ' then is ', encoder_outputs.last_hidden_state.shape[0], ' pivot is ', pivot_index, ' previous shape is ', input_ids_list[i-1].shape)
        #print('first len is ',encoder_outputs.last_hidden_state.shape[0], ' second len is ', len(pivot_index))
            hidden_embedding += encoder_outputs.last_hidden_state[np.arange(encoder_outputs.last_hidden_state.shape[0]),pivot_index,:].unsqueeze(1)
            #print('hidden embedding shape is ', hidden_embedding.shape, ' hidden_states shape is ', hidden_states.shape)
            #print(' hidden_states is ', hidden_states)
            updated_input = torch.cat((hidden_embedding, hidden_states), dim=1).contiguous()
            #print('updated_input dimension is ', hidden_states)
            final_hidden_states = self.decoder(inputs_embeds=updated_input)[0]
            logits = self.lm_head(final_hidden_states)
        #logits = F.log_softmax(logits, dim=-1)
            shifted_prediction_scores = logits[:, 1:-1, :]
        
            if labels_list == None:
                return {'logits':logits}

            labels_list[i][attention_mask_list[i] == 0] = -100
            pivot_twodim_decoder = ((input_ids_list[i] == self.config.eos_token_id).cumsum(1) == 1)
            labels_list[i][pivot_twodim_decoder] = self.config.eos_token_id
        #print('labels are ', labels[0][])
            labels_list[i] = labels_list[i][:, 1:]
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.contiguous().view(-1, self.config.vocab_size), labels_list[i].contiguous().view(-1))
            total_loss += lm_loss
            logits_pack.append(logits[:,1:,:])
        #print('im successfully here')
        return {'loss': total_loss}#, 'logits_pack': logits_pack}#, 'logits_pack':logits_pack}