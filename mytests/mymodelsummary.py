from transformers import PreTrainedModel, GPT2Model, GPT2LMHeadModel, GPT2Config, DataCollatorForLanguageModeling, AutoTokenizer
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
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)


    def append_and_return(self, lst, element):
        lst.append(element)
        return lst

    def forward(self, input_ids, labels=None, attention_mask=None):
        separation_indices = (input_ids == -1).nonzero()[:,1]
        #print('shape[0] is ', input_ids.shape[0])
        #print('separation-indices are ', separation_indices)
        input_ids_main_obj =  [self.append_and_return(input_ids[i , :separation_indices[i]].tolist(), self.config.eos_token_id) for i in range(input_ids.shape[0])]                    
        input_ids_summary = [self.append_and_return(input_ids[i , separation_indices[i] + 1:].tolist(), self.config.eos_token_id) for i in range(input_ids.shape[0])] 
        #print('input_ids_main is ', input_ids_main, ' input_ids_summary is ', input_ids_summary)
        input_ids_main = self.collator(input_ids_main_obj)['input_ids'].to('cuda')
        input_ids_summary = self.collator(input_ids_summary)
        labels = input_ids_summary['labels'].to('cuda')
        input_ids_summary = input_ids_summary['input_ids'].to('cuda')
        #print('input_ids_main is ', input_ids_main, ' input_ids_summary is ', input_ids_summary)
        
        
        
        pivot_twodim_summary = ((input_ids_summary == self.config.eos_token_id).cumsum(1) == 1)
        #pivot_index_summary = pivot_twodim_summary.nonzero()[:,1]
        pivot_twodim_main = ((input_ids_main == self.config.eos_token_id).cumsum(1) == 1)
        pivot_index_main = pivot_twodim_main.nonzero()[:,1]
       

        position_ids = torch.arange(input_ids_summary.shape[1], dtype=torch.long, device='cuda')
        position_ids = position_ids.unsqueeze(0)
        inputs_embeds = self.decoder.wte(input_ids_summary)
        position_embeds = self.decoder.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        encoder_outputs = self.encoder(input_ids_main)
        # this line is assuming there is a eos token at the end of the first story
        #hidden_embedding = encoder_outputs.last_hidden_state[:,-1,:].unsqueeze(1)
        
        try:
            hidden_embedding = encoder_outputs.last_hidden_state[np.arange(encoder_outputs.last_hidden_state.shape[0]),pivot_index_main,:].unsqueeze(1)
        except Exception as e:
            print('excption is ', e)
            print('obj is ', self.collator(input_ids_main_obj)['input_ids'].shape)
            print('encoder_outputs.last_hidden_state.shape[0] is ',encoder_outputs.last_hidden_state.shape[0], ' len pivot_index_main is ',pivot_index_main
              , ' shape encoder_outputs.last_hidden_state is ', encoder_outputs.last_hidden_state.shape)
            print('salam')
        updated_input = torch.cat((hidden_embedding, hidden_states), dim=1)
        final_hidden_states = self.decoder(inputs_embeds=updated_input)[0]
        logits = self.lm_head(final_hidden_states)
        shifted_prediction_scores = logits[:, 1:-1, :]
        
        #if labels == None:
        #    return {'logits':logits}
        #labels = input_ids_summary
        labels[input_ids_summary == self.config.eos_token_id] = -100
        ## count the first eos token in calculating the loss
        labels[pivot_twodim_summary] = self.config.eos_token_id
        labels = labels[:, 1:]
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(shifted_prediction_scores.contiguous().view(-1, self.config.vocab_size), labels.contiguous().view(-1))
        return {'loss': lm_loss, 'logits':logits[:,1:,:]}
        