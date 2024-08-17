## retriving the hidden state
from transformers import AutoTokenizer, AutoModel
import torch
tokenizer = AutoTokenizer.from_pretrained('gpt2')
input_str = 'Once upon a time,'
#input_str = ''
#input_ids = tokenizer(input_str, return_tensors='pt')['input_ids'].to('cuda')
input_ids = tokenizer(input_str, return_tensors='pt')['input_ids'].to('cuda')
input_ids = input_ids[:,5:]
#print('input_ids is ', input_ids)

print('input ids are ', input_ids)
from mymodelsummary import MyModel 
#mymodel = MyModel.from_pretrained('nodotmodel3_weights_sirui')
mymodel = MyModel.from_pretrained('nodotmodel3_weights_2024-08-09--02:19:44alaki') ## second summary model
#mymodel = MyModel.from_pretrained('nodotmodel3_weights_2024-08-05--20:10:40alaki') ## final correct with eos and not dot 
#mymodel = MyModel.from_pretrained('posmodel3_weights_2024-08-01--05:36:40alaki')
mymodel = mymodel.to('cuda')
mymodel.eval()

import sys
raw_story = ''' Once upon a time, there was a kind farmer. He had a big cow. The cow
was sad. The farmer did not know why. One day, a little boy came to
the farm. He saw the sad cow. The boy kneeled down to talk to the cow.
"Why are you sad, cow?" he asked. The cow said, "I am lonely. I want a
friend." The kind farmer heard the cow. He wanted to help. So, he got
another cow to be friends with the sad cow. The sad cow was happy now.
They played together every day. And the kind farmer, the little boy,
and the two cows all lived happily ever after.<|endoftext|>'''

if len(sys.argv) > 1:
    input_string = sys.argv[1]
    print('string is provided')
    raw_story = input_string
else:
    print('No string provided')

from datasets import load_dataset
tinystories = load_dataset('roneneldan/TinyStories')
raw_story2 = tinystories['validation']['text'][1010]
raw_story2 = raw_story2 + '<|endoftext|>'

raw_input_ids = tokenizer(raw_story, return_tensors='pt')['input_ids']
raw_input_ids2 = tokenizer(raw_story2, return_tensors='pt')['input_ids']
## add the eos token at the end
#raw_input_ids = torch.cat((raw_input_ids, torch.tensor([[tokenizer.eos_token_id]])), dim=1)
raw_input_ids = raw_input_ids.to('cuda')
raw_input_ids2 = raw_input_ids2.to('cuda')
#print('raw input ids is ', raw_input_ids.type)
hidden_embedding = mymodel.encoder(raw_input_ids).last_hidden_state.detach()[:, -1, :].unsqueeze(1)
hidden_embedding2 = mymodel.encoder(raw_input_ids2).last_hidden_state.detach()[:, -1, :].unsqueeze(1)
hidden_embedding = (hidden_embedding2 + hidden_embedding2)/2


print(hidden_embedding.shape)

## generate the story starting from once upon a time


#mymodel = AutoModel.from_pretrained('gpt2').to('cuda')
#input_ids = input_ids[0]
for i in range(300):
    if i == 0:
        updated_input_ids = hidden_embedding
        print('hidden embedding size is ', hidden_embedding.shape)
    else:   
        #print(mymodel(input_ids)['logits'])
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device='cuda')
        position_ids = position_ids.unsqueeze(0)
        inputs_embeds = mymodel.decoder.wte(input_ids)
        position_embeds = mymodel.decoder.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        updated_input_ids = torch.cat((hidden_embedding, hidden_states), dim=1)

    output_index = torch.argmax(mymodel.lm_head(mymodel.decoder(inputs_embeds=updated_input_ids)[0]), dim=2)
    #print('output index is ', output_index[:,-1])
    #print('input_ids is ', input_ids)
    print('im here!')
    input_ids = torch.cat((input_ids, output_index[:,-1].unsqueeze(0)), dim=1)
    print('input_ids shape is ', input_ids.shape)
    print('i passed here!')
    print('input_ids is ', input_ids.shape)
    #print(input_ids)
print('INPUT:', raw_story2)
print('OUTPUT: ', tokenizer.decode(input_ids[0]))
#print(mymodel.lm_head.weight)
#print(mymodel(input_ids.to('cuda')))

