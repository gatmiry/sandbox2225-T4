from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, AutoTokenizer, DataCollatorForLanguageModeling, GPT2Tokenizer

args = TrainingArguments(
    output_dir='./alakioutputs',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy='steps',
    logging_strategy='steps',
    eval_steps=100,
    logging_steps=50,
    save_steps=100,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    #report_to='wandb'
)
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
from datasets import load_from_disk
import numpy as np
tokenized_datasets = load_from_disk('./model3-outputs')
tokenized_dataets= {
    'train':tokenized_datasets['train'].select(np.arange(100)),
    'validation':tokenized_datasets['validation'].select(np.arange(100)),
}

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)
trainer.train()