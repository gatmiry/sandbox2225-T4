## retriving the hidden state
################################################################# This file is out-dated
from transformers import AutoTokenizer, AutoModel
import torch
tokenizer = AutoTokenizer.from_pretrained('gpt2')
input_str = 'Once upon a time,'
input_ids = tokenizer(input_str, return_tensors='pt')['input_ids'].to('cuda')
from mymodelnew2 import MyModel 
mymodel = MyModel.from_pretrained('posmodel3_weights_2024-07-29--17:41:44alaki')
mymodel = mymodel.to('cuda')
mymodel.eval()
print(input_ids)

import sys
raw_story = '''Once upon a time, there was a kind farmer. He had a big cow. The cow
was sad. The farmer did not know why. One day, a little boy came to
the farm. He saw the sad cow. The boy kneeled down to talk to the cow.
"Why are you sad, cow?" he asked. The cow said, "I am lonely. I want a
friend." The kind farmer heard the cow. He wanted to help. So, he got
another cow to be friends with the sad cow. The sad cow was happy now.
They played together every day. And the kind farmer, the little boy,
and the two cows all lived happily ever after.</s>Once upon a time, there was a girl named Mia. Mia loved her jewelry.
She had a big box full of pretty things. She liked to wear them all
day. But at night, she had to sleep. One day, Mia met a talking cat
named Tom. Tom was a tough cat, but he was nice. Tom said, "Hi, Mia! I
like your jewelry. Can I wear some too?" Mia said, "Yes, Tom. You can
wear my jewelry, but we have to give it back before we sleep." So, Mia
and Tom played together. They wore the jewelry and had fun. They
pretended to be kings and queens. They laughed and danced. But soon,
the sun went down, and it was time for bed. Mia said, "Tom, we must
give back the jewelry now. It's time to sleep." Tom gave back the
jewelry and said, "Thank you, Mia. I had fun today." They put the
jewelry back in the box and went to sleep. Mia and Tom were happy, and
they had sweet dreams.
'''

if len(sys.argv) > 1:
    input_string = sys.argv[1]
    print('string is provided')
    raw_story = input_string
else:
    print('No string provided')

from datasets import load_dataset
tinystories = load_dataset('roneneldan/TinyStories')
#raw_story = tinystories['validation']['text'][1010]

raw_input_ids = tokenizer(raw_story, return_tensors='pt')['input_ids']
## add the eos token at the end
#raw_input_ids = torch.cat((raw_input_ids, torch.tensor([[tokenizer.eos_token_id]])), dim=1)
raw_input_ids = raw_input_ids.to('cuda')
print('raw input ids is ', raw_input_ids.type)
hidden_embedding = mymodel.encoder(raw_input_ids).last_hidden_state.detach()[:, -1, :].unsqueeze(1)
print(hidden_embedding.shape)

## generate the story starting from once upon a time


#mymodel = AutoModel.from_pretrained('gpt2').to('cuda')
#input_ids = input_ids[0]
for i in range(500):
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
    input_ids = torch.cat((input_ids, output_index[:,-1].unsqueeze(0)), dim=1)
    #print(input_ids)
print('INPUT: ', raw_story)
print('OUTPUT: ', tokenizer.decode(input_ids[0]))
#print(mymodel.lm_head.weight)
#print(mymodel(input_ids.to('cuda')))

