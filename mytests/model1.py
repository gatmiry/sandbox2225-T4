from transformers import AutoTokenizer, T5EncoderModel, AutoConfig
import torch
from transformers import AutoModel

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
encoder_model = T5EncoderModel.from_pretrained("google-t5/t5-small")
input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids
print('input_ids type is ', input_ids.size())
input_ids_two = tokenizer("Studies have been shown that owning a dog is ", return_tensors="pt").input_ids
#print('input_ids are ', input_ids)
encoded_outputs = encoder_model(input_ids=input_ids)
last_hidden_states = encoded_outputs.last_hidden_state
last_hidden_state = last_hidden_states[0,:,-1]
updated_input_ids = [torch.cat((last_hidden_state, item), dim=0) for item in input_ids_two]
#print('updated input ids is ', updated_input_ids[0])

#print('last_hidden_state is ',last_hidden_state)
#print('input_ids[0] is ', input_ids[0])
#print('concatenated vector is ', torch.cat((last_hidden_state, input_ids[0]), dim=0))
#print("decoded input is ", input_ids.shape)

## second phase, feeding the decoder
#model = AutoModel.from_pretrained('gpt2')
config = AutoConfig.from_pretrained('gpt2')
config.vocab_size = 1000
model = AutoModel.from_config(config)

#output = model.generate(inputs, )