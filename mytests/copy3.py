from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5EncoderModel

# Initialize tokenizer and model
model_name = 'google-t5/t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
encoder = T5EncoderModel.from_pretrained(model_name)
decoder = T5ForConditionalGeneration.from_pretrained(model_name).decoder

# Example input text
input_text = "Translate this sentence into French: Hello, how are you?"

# Tokenize input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Encode input text with T5 encoder
encoder_outputs = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

# Example of using encoder outputs
last_hidden_state = encoder_outputs.last_hidden_state
# Use last_hidden_states as needed, such as for downstream tasks or additional processing

# Example of using decoder separately for generation
# Generate text based on a prompt using the decoder
prompt = "Translate: English to French"
decoder_outputs = decoder(input_ids=inputs['input_ids'], encoder_hidden_states=last_hidden_state, attention_mask=inputs['attention_mask'])

# Decode generated output


decoded_output = tokenizer.decode(decoder_outputs[0][0][0], skip_special_tokens=False)
print("Decoded Output:", decoder_outputs[0][0][0])