from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig, T5EncoderModel, AutoTokenizer, GPT2LMHeadModel, T5Config
import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss
#from timm.models.resnet import BasicBlock, Bottleneck, ResNet

class MyConfig(PretrainedConfig):
    model_type = 'my_model'

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

class MyModel(PreTrainedModel):
    config_class = MyConfig

    def __init__(self, config):
        super().__init__(config)
        #self.encoder_config = AutoConfig.from_pretrained('google-t5/t5-small')
        #self.decoder_config = AutoConfig.from_pretrained('gpt2')
        #decoder_config = AutoConfig.from_pretrained("gpt2")
        #self.encoder = T5EncoderModel.from_pretrained('google-t5/t5-small')
        config.max_position_embeddings = 512
        config.embd_pdrop = 0.1
        config.n_inner = None
        config.scale_attn_weights = True
        config.scale_attn_by_inverse_layer_idx = False
        config.reorder_and_upcast_attn = False
        config.attn_pdrop = 0.1
        config.resid_pdrop = 0.1
        config2 = AutoConfig.from_pretrained('gpt2')
        print('gpt2 config has ', config2.activation_function)
        encoder_config = AutoConfig.from_pretrained('gpt2')
        decoder_config = AutoConfig.from_pretrained('google-t5/t5-small')
        decoder_tmp = GPT2LMHeadModel.from_pretrained('google-t5/t5-small')
        print('decoder_tmp max_position_embeddings is ', decoder_tmp.config.resid_pdrop)
        
        #decoder_config.
        #print('encoder type is ', encoder_config.type)
        config2 = T5Config(
            vocab_size=32128,  # Define your vocabulary size based on your data
            d_model=512,       # Define the model size (hidden layer size)
            num_layers=6,      # Define the number of layers in the encoder
            num_heads=8,       # Define the number of attention heads
            d_ff=2048,         # Define the size of the feed-forward layers
            initializer_range=0.02  # Set the range for weight initialization
        )

        #print('config is ', encoder_config)
        #print('config hidden size is ', config.hidden_size)
        #encoder_config.hidden_size = config.hidden_size
        
        self.encoder = T5EncoderModel(config)
        #print('shared weights is ', self.encoder.config.shared.weight)
        #self.second_encoder = T5EncoderModel(config)
        #print('this is ', self.second_encoder.shared.weight.shape)
        self.decoder = GPT2LMHeadModel(config)
        #self.decoder = AutoModel.from_config(self.decoder_config)

    def forward(self, tensor, labels=None, attention_mask=None):
        encoder_outputs = self.encoder(tensor)
        #print('last hidden state is ', encoder_outputs.last_hidden_state.shape)
        hidden_embedding = encoder_outputs.last_hidden_state[:,-1,:]
        # just to obtain the hidden embeddings
        print('hidden size asli is ', self.encoder.config.hidden_size)
        with torch.no_grad():
            decoder_hidden_states = self.second_encoder(tensor, output_hidden_states=True).hidden_states
            print('decoder shape is ', decoder_hidden_states[0].shape)
        hidden_embedding_dim = hidden_embedding.shape[1]
        updated_input = torch.cat((hidden_embedding, tensor), dim=1)
        print('updated input shape is ', updated_input.shape)
        
        #print('encoder outputs is ', encoder_outputs)
        logits = self.decoder(inputs_embeds=torch.ones(1,1,768))['logits']
        logits = F.log_softmax(logits, dim=-1)
        #print('logits shape is ', logits.shape)
        shifted_prediction_scores = logits[:, :-1, :]
        print('size is ', shifted_prediction_scores.shape)
        labels[attention_mask == 0] = -100 
        labels = labels[:, 1:]
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        return {'loss': lm_loss, 'logits':logits}

#myconfig = MyConfig()
config_name = 'google-t5/t5-small'
myconfig = AutoConfig.from_pretrained('gpt2')
#myconfig.hidden_size = 768
mymodel = MyModel(myconfig)
#decoder_model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')
print('hidden size is ', myconfig.hidden_size)
inputs = tokenizer('Studies have been shown that owning a dog is good for your health', return_tensors='pt')
outputs = mymodel(inputs['input_ids'], labels=inputs['input_ids'], attention_mask=inputs['attention_mask'])
print('output loss is ', outputs['loss'])
#print('output-ids is ', output_ids.shape)
#print('output is ', tokenizer.batch_decode(output_ids, skip_special_tokens=True))