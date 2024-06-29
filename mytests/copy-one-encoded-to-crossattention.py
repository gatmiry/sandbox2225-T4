from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

input_text = 'Translate this sentence into French: Hello, how are you?'
inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)

outputs = model.generate(input_ids=inputs['input_ids'], max_length=20)


#encoder_outputs = model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
#decoder_outputs = model.decoder(input_ids=model._shift_right(inputs['input_ids']),  
                                #encoder_hidden_states=encoder_outputs.last_hidden_state)
                                #encoder_attention_mask=inputs['attention_mask'])


#print('decoder_ouputs is ', decoder_outputs[0].shape)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))