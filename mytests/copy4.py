from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel

# Initialize tokenizer and model
model_name = 'google-t5/t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
encoder = T5EncoderModel.from_pretrained(model_name)
decoder = T5ForConditionalGeneration.from_pretrained(model_name)

# Example input text
input_text = "Translate this sentence into French: Hello, how are you?"
input_text2 = "Translate this sentence into French: How can I go the third avenue?"

# Tokenize input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
inputs2 = tokenizer(input_text2, return_tensors="pt", padding=True, truncation=True)

# Encode input text with T5 encoder
encoder_outputs = encoder(input_ids=inputs2['input_ids'], attention_mask=inputs2['attention_mask'])
print('encoder output is ', type(encoder_outputs))

# Example of using encoder outputs
last_hidden_state = encoder_outputs.last_hidden_state
prompt = "Translate: English to French"
decoder_outputs = decoder.generate(input_ids=inputs2['input_ids'], encoder_outputs=encoder_outputs)

# Decode generated output


decoded_output = tokenizer.decode(decoder_outputs[0], skip_special_tokens=True)
print("Decoded Output:", decoded_output)