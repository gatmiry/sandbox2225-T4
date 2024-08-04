from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig, T5EncoderModel, AutoTokenizer, GPT2LMHeadModel, T5Config, GPT2Model, DataCollatorForLanguageModeling, GPT2Config
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torch.nn import CrossEntropyLoss
# loading the dataset
from datasets import load_dataset, DatasetDict
from mymodel import MyModel

myconfig = AutoConfig.from_pretrained('gpt2')
context_length=512
myconfig.n_ctx = context_length
#myconfig.hidden_size = 768

## initializing the model
mymodel = MyModel(myconfig)
#mymodel.set_config(myconfig)
tokenizer =  AutoTokenizer.from_pretrained('gpt2')
#inputs = tokenizer('Studies have been shown that owning a dog is good for your health', return_tensors='pt')
#outputs = mymodel(inputs['input_ids'], labels=inputs['input_ids'], attention_mask=inputs['attention_mask'])
#rint('output loss is ', outputs['loss'])

from datasets import load_from_disk
tokenized_datasets = load_from_disk('/mnt/t-kgatmiry-data/model3-outputs-gpt2tokenizer')
## slicing the dataset to a smaller size
tokenized_datasets = {
    'train': tokenized_datasets['train'].select(np.arange(100)),#500000
    'validation': tokenized_datasets['validation'].select(np.arange(100))#5000
}


## now instantiate the data collator
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# now set the training arguments
from transformers import Trainer, TrainingArguments
import datetime
datetime_str = datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S')

args = TrainingArguments(
    output_dir='/mnt/t-kgatmiry-output/nnmodel3_default_outputs_' + datetime_str,
    logging_dir='/mnt/t-kgatmiry-output/nnmodel3_default_logs_' + datetime_str,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    evaluation_strategy='steps',
    logging_strategy='steps',
    eval_steps=100,
    logging_steps=100,
    save_steps=1000,
    gradient_accumulation_steps=8,
    num_train_epochs=8,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    fp16=True,
    #deepspeed='./ds_config.json'
)

trainer = Trainer(model=mymodel,
                  tokenizer=tokenizer,
                  args=args,
                  data_collator=data_collator,
                  train_dataset=tokenized_datasets['train'],
                  eval_dataset=tokenized_datasets['validation']
                )
num_gpus_torch = torch.cuda.device_count()
#import deepspeed
#num_gpus_deep = deepspeed.comm.get_world_size()
print('number if gpus being used is with deepspeed is ', num_gpus_torch)
trainer.train()
model_state_dict = mymodel.state_dict()
torch.save(model_state_dict, '/mnt/t-kgatmiry-output/nnmodel3_weights_' + datetime_str)
log_history_file = '/mnt/t-kgatmiry-output/nnmodel3_logs_' + datetime_str + '.json'
import json
with open(log_history_file, 'w') as f:
    json.dump(trainer.state.log_history, f, indent=4)

