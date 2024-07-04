from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig, T5EncoderModel, AutoTokenizer, GPT2LMHeadModel, T5Config, GPT2Model, DataCollatorForLanguageModeling, GPT2Config
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torch.nn import CrossEntropyLoss
# loading the dataset
from datasets import load_dataset, DatasetDict

#from timm.models.resnet import BasicBlock, Bottleneck, ResNet

class MyConfig(PretrainedConfig):
    model_type = 'my_model'

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

class MyModel(PreTrainedModel):
    config_class = GPT2Config

    def __init__(self, config):
        super().__init__(config)
        #self.encoder = T5EncoderModel.from_pretrained('google-t5/t5-small')
        
        
        self.encoder = GPT2Model(config)
        self.second_encoder = GPT2Model(config)
        self.decoder = GPT2LMHeadModel(config)

    def forward(self, input_ids, labels=None, attention_mask=None):
        encoder_outputs = self.encoder(input_ids)
        #print('last hidden state is ', encoder_outputs.last_hidden_state.shape)
        hidden_embedding = encoder_outputs.last_hidden_state[:,-1,:].unsqueeze(1)
        # just to obtain the hidden embeddings
        with torch.no_grad():
            decoder_hidden_inputs = self.second_encoder(input_ids, output_hidden_states=True).hidden_states[0]
            #print('decoder shape is ', decoder_hidden_inputs.shape)
        #hidden_embedding_dim = hidden_embedding.shape[2]
        updated_input = torch.cat((hidden_embedding, decoder_hidden_inputs), dim=1)
        
        
        #print('encoder outputs is ', encoder_outputs)
        logits = self.decoder(inputs_embeds=updated_input)['logits']
        logits = F.log_softmax(logits, dim=-1)
        #print('logits shape is ', logits.shape)
        shifted_prediction_scores = logits[:, 1:-1, :]
        #print("shifted_prediction_scores shape is ", shifted_prediction_scores.shape)
        labels[attention_mask == 0] = -100 
        labels = labels[:, 1:]
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(shifted_prediction_scores.contiguous().view(-1, self.config.vocab_size), labels.contiguous().view(-1))
        return {'loss': lm_loss, 'logits':logits[:,1:,:]}

#myconfig = MyConfig()
myconfig = AutoConfig.from_pretrained('gpt2')
context_length=512
myconfig.n_ctx = context_length
#myconfig.hidden_size = 768

## initializing the model
mymodel = MyModel(myconfig)
#mymodel = MyModel.from_pretrained('./model3weights/')

#decoder_model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')
inputs = tokenizer('Studies have been shown that owning a dog is good for your health', return_tensors='pt')
outputs = mymodel(inputs['input_ids'], labels=inputs['input_ids'], attention_mask=inputs['attention_mask'])
print('output loss is ', outputs['loss'])
#print('output-ids is ', output_ids.shape)
#print('output is ', tokenizer.batch_decode(output_ids, skip_special_tokens=True))

# loading the dataset


def tokenize(element):
    #print('element is ', element.keys())
    #return {'input_ids': []}
    outputs = tokenizer(
        element['text'],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_batch = []
    for length, input_ids in zip(outputs['length'], outputs['input_ids']):
        if length == context_length:
            input_batch.append(input_ids)
    return {'input_ids': input_batch}

#ds = load_dataset("roneneldan/TinyStories")
#tokenized_datasets = ds.map(
#    tokenize, batched=True, remove_columns=ds["train"].column_names
#)
from datasets import load_from_disk
tokenized_datasets = load_from_disk('./model3-outputs')
print('this is ', len(tokenized_datasets['train'][0]['input_ids']))
## slicing the dataset to a smaller size
tokenized_datasets = {
    'train': tokenized_datasets['train'].select(np.arange(1000)),#500000
    'validation': tokenized_datasets['validation'].select(np.arange(1000))#5000
}

#print('tokenized_datasets is ', tokenized_datasets)


#print('ds is ', tokenizer(ds['train']['text'][0], return_length=True))
#subdataset = DatasetDict(
#    {
#        'train': ds['train'].select(np.arange(10))
#    }
#)
#print('subset is ', subset['text'])


## now instantiate the data collator
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)



#out = data_collator([tokenized_datasets['train'][i] for i in range(5)])

# now set the training arguments
from transformers import Trainer, TrainingArguments
import datetime
datetime_str = datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S')

args = TrainingArguments(
    output_dir='testmodel3outputs_' + datetime_str,
    logging_dir='./testmodel3logs_' + datetime_str,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy='steps',
    logging_strategy='steps',
    eval_steps=10,
    logging_steps=1,
    save_steps=1,
    gradient_accumulation_steps=8,
    num_train_epochs=4,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    fp16=True,
)

print('dataset is ', tokenized_datasets)
#print('config is ', mymodel.config)


#from transformers import GPT2LMHeadModel
#config = AutoConfig.from_pretrained(
#    "gpt2",
#    vocab_size=len(tokenizer),
#    n_ctx=context_length,
#    bos_token_id=tokenizer.bos_token_id,
#    eos_token_id=tokenizer.eos_token_id,
#)
#test_model = GPT2LMHeadModel(config)


trainer = Trainer(model=mymodel,
                  tokenizer=tokenizer,
                  args=args,
                  data_collator=data_collator,
                  train_dataset=tokenized_datasets['train'],
                  eval_dataset=tokenized_datasets['validation']
                )

trainer.train()
trainer.save_model('./testmodel3weights_' + datetime_str)
log_history_file = './testmodel3logs_' + datetime_str + '.json'
import json
with open(log_history_file, 'w') as f:
    json.dump(trainer.state.log_history, f, indent=4)
#mymodel.save_pretrained('./model4/mymodel', from_pt=True)