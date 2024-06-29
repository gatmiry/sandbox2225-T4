from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilgpt2')
from datasets import load_dataset
eli5 = load_dataset("eli5_category", split="train[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)
#print('eli5 is ', eli5)
#print(eli5['train'][0])
eli5 = eli5.flatten()
print('preeli is ', eli5)

def preprocess_function(examples):
    #print('examples is ', examples)
    return tokenizer([" ".join(x) for x in examples["answers.text"]])
tokenized_eli5 = eli5.map(
    preprocess_function, 
    batched=True,
    num_proc=4,
    remove_columns=eli5['train'].column_names,
)

#print(tokenized_eli5['train']['input_ids'][0])

block_size = 128
def group_texts(examples):
    #print('examples is ', examples[])
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0,total_length, block_size)]
        for k,t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result

lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
#print('lm_dataset is ', len(lm_dataset['train']['input_ids']))
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

##################### Now it's time for training!
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
model = AutoModelForCausalLM.from_pretrained('distilbert/distilgpt2')

training_args = TrainingArguments(
    output_dir='my_awesome_eli5_clm-model',
    eval_strategy='epoch',
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset['train'],
    eval_dataset=lm_dataset['test'],
    data_collator=data_collator
)

trainer.train()