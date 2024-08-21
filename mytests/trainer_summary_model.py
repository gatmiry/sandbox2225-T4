from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig, T5EncoderModel, AutoTokenizer, GPT2LMHeadModel, T5Config, GPT2Model, DataCollatorForLanguageModeling, GPT2Config
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torch.nn import CrossEntropyLoss
# loading the dataset
from datasets import load_dataset, DatasetDict
from mymodelsummary import MyModel
#from mycollators import DataCollatorForTextSimilarity


myconfig = AutoConfig.from_pretrained('gpt2')
context_length=2048
myconfig.n_ctx = context_length
myconfig.max_position_embeddings = context_length
#myconfig.hidden_size = 768

## initializing with the pretrained summary model 
mymodel = MyModel(myconfig)
#mymodel = MyModel.from_pretrained('nodotmodel3_weights_2024-08-05--20:10:40alaki')
#mymodel = MyModel(myconfig)
#mymodel.set_config(myconfig)
tokenizer =  AutoTokenizer.from_pretrained('gpt2')
#inputs = tokenizer('Studies have been shown that owning a dog is good for your health', return_tensors='pt')
#outputs = mymodel(inputs['input_ids'], labels=inputs['input_ids'], attention_mask=inputs['attention_mask'])
#rint('output loss is ', outputs['loss'])

from datasets import load_from_disk
#tokenized_datasets = load_from_disk('./model3-outputs-gpt2tokenizer-varlength')
tokenized_datasets = load_from_disk('./tinystories-withsummary-gpt2tokenizer-49')
#tokenized_datasets = load_from_disk('./model3-outputs-gpt2tokenizer-askubuntubody-150000train-10000test')
#tokenized_datasets = load_from_disk('./model3-outputs-gpt2tokenizer-askubuntubody-margintraining-150000train-10000test')

## slicing the dataset to a smaller size

#tokenized_datasets = {
#    'train': tokenized_datasets['train'].select(np.arange(500)),#500000
#    'validation': tokenized_datasets['validation'].select(np.arange(200))#5000
#}


## now instantiate the data collator
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
#data_collator = DataCollatorForTextSimilarity(tokenizer=tokenizer, sub_batch_size=300)

# now set the training arguments
from transformers import Trainer, TrainingArguments
import datetime
datetime_str = datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S') + 'alaki'

args = TrainingArguments(
    output_dir='nodotmodel3_default_outputs_' + datetime_str,
    logging_dir='./nodotmodel3_default_logs_' + datetime_str,
    per_device_train_batch_size=30,
    per_device_eval_batch_size=30,
    evaluation_strategy='steps',
    logging_strategy='steps',
    eval_steps=500,
    logging_steps=100,
    save_steps=1000000,
    gradient_accumulation_steps=8,
    num_train_epochs=4,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    fp16=True,
    remove_unused_columns=False
)

trainer = Trainer(model=mymodel,
                  tokenizer=tokenizer,
                  args=args,
                  data_collator=data_collator,
                  train_dataset=tokenized_datasets['train'].select(np.arange(1,4857871)),
                  eval_dataset=tokenized_datasets['train'].select(np.arange(4857872, 4867871))
                )
#print('model tied weight is ', mymodel._tied_weights_keys)
num_gpus_torch = torch.cuda.device_count()
print('number if gpus being used is with deepspeed is ', num_gpus_torch)
print('number of huggingface gpus ', trainer.args._n_gpu)
trainer.train()

#model_state_dict = mymodel.state_dict()
#torch.save(model_state_dict, './nnmodel3_weights_' + datetime_str)
#from safetensors.torch import save_model
#save_model(mymodel, './safemodel3_weights_' + datetime_str)
trainer.save_model('./nodotmodel3_weights_' + datetime_str)
#print('model tied weight is ', mymodel._tied_weights_keys)
log_history_file = './nodotmodel3_logs_' + datetime_str + '.json'
import json
with open(log_history_file, 'w') as f:
    json.dump(trainer.state.log_history, f, indent=4)





#mystory = ''' Once upon a time, there was a thoughtful girl named Sue. Sue loved tohelp her mom around the house. One day, her mom asked her to wipe thetable after they ate their lunch. Sue was happy to help. As Sue waswiping the table, she saw a pretty candle on the window sill. Thecandle was her mom's favorite. Sue wanted to do something nice for hermom, so she said, "Mom, can I light the candle for you?" Her mom said,"Yes, but be very careful." Sue carefully lit the candle and put it onthe table. Her mom was so happy to see the pretty candle. They bothsat and watched the candle burn. Sue's mom said, "Thank you, Sue, forbeing so thoughtful and careful." Sue felt proud that she could helpher mom. The moral of the story is to always be thoughtful and carefulwhen helping others.CLOSE TEXT IS:  Once upon a time, there was a little girl named Mia. She loved to helpher mom in the kitchen. One day, her mom asked her to unpack a bigbox. Mia was very happy to help. Inside the box, there was a newfaucet for the kitchen sink. Mia's mom said, "This faucet is veryspecial. It can go high and low." Mia watched as her mom put the newfaucet on the sink. When it was high, the water came out fast. When itwas low, the water came out slow. Mia learned that helping her mom wasfun and important. She felt proud when she saw the new faucet working.The moral of the story is that helping others is a good thing to do.'''
#outputs = tokenizer(mystory, return_tensors='pt')
#mymodel.eval()
#print(mymodel(input_ids=outputs['input_ids'].cuda(), labels=outputs['input_ids'].cuda())['loss'])
