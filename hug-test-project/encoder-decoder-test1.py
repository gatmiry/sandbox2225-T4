from transformers import AutoTokenizer, T5EncoderModel

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5EncoderModel.from_pretrained("google-t5/t5-small")

input_ids = tokenizer("Studies have shown that owning a dog is good for you", return_tensors='pt').input_ids
outputs = model(input_ids=input_ids, output_hidden_states=True)

print('encoder output is ', tokenizer.decode(outputs.last_hidden_state[0][0], skip_special_tokens=True))