## loading askubuntu
import pandas as pd
import numpy as np
df = pd.read_csv("texts_raw_fixed.txt", sep="\t", header=None)
df.columns = ['qid', 'title', 'body']
from datasets import Dataset, DatasetDict
dataset = Dataset.from_pandas(df)
ds = DatasetDict({
    'train': dataset.remove_columns(['qid']).select(range(150000)),
    'validation': (dataset.remove_columns(['qid'])).select(range(150001, 160000)),
})


## defining tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

context_length = 512
eos_id = tokenizer.eos_token_id#tokenizer(tokenizer.eos_token)['input_ids'][0]
istring = np.vectorize(lambda x: isinstance(x, str))
def tokenize(element):
    #count = 0
    #for text in element['body']:
    #    if isinstance(text, str) == False:
    #        print('THIS INPUT IS NOT STRING!!!')
    #        print('text is ', text, ' element body is ', element['body'][count -1: count +1])
    #    count += 1
    
    element['body'] = list(np.array(element['body'])[istring(np.array(element['body']))])
    outputs = tokenizer(
        element['body'],
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
tokenized_datasets.save_to_disk('./model3-outputs-gpt2tokenizer-askubuntubody-150000train-10000test')