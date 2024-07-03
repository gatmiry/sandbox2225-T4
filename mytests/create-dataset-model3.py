## loading tinystories
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")

## defining tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')

context_length = 512
def tokenize(element):
    print('element is ', len(element['text']))
    #return {'input_ids': []}
    #print('len is ', ('</s>'.join(x) for x in element['text']).type)
    outputs = tokenizer(
        ['</s>'.join(element['text'])],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_batch = []
    for length, input_ids in zip(outputs['length'], outputs['input_ids']):
        #print('length is ', length)
        if length == context_length:
            input_batch.append(input_ids)
    #print('batch length is ', len(input_batch))
    return {'input_ids': input_batch}

#tokenizing the dataset
tokenized_datasets = ds.map(
    tokenize, batched=True, remove_columns=ds["train"].column_names
)
tokenized_datasets.save_to_disk('./model3-outputs')