## loading tinystories
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")

## defining tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

context_length = 512
eos_id = tokenizer.eos_token_id#tokenizer(tokenizer.eos_token)['input_ids'][0]

def tokenize(element):
    print('element is ', len(element['text']))
    #return {'input_ids': []}
    #print('len is ', ('</s>'.join(x) for x in element['text']).type)
    outputs = tokenizer(
        element['text'],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_batch = []
    for length, input_ids in zip(outputs['length'], outputs['input_ids']):
        #print('length is ', length)
        if length <= context_length:
            input_ids.append(eos_id)
            #while len(input_ids) < context_length:
            #    input_ids.append(eos_id)
            input_batch.append(input_ids)
    #print('batch length is ', len(input_batch))
    return {'input_ids': input_batch}

#tokenizing the dataset
tokenized_datasets = ds.map(
    tokenize, batched=True, remove_columns=ds["train"].column_names
)
tokenized_datasets.save_to_disk('./model3-outputs-gpt2tokenizer-varlength')