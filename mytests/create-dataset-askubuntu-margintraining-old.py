## loading askubuntu
import pandas as pd
import numpy as np
df = pd.read_csv("texts_raw_fixed.txt", sep="\t", header=None)
df.columns = ['qid', 'title', 'body']
from datasets import Dataset, DatasetDict
dataset = Dataset.from_pandas(df)
similar_train_df = pd.read_csv('train_random.txt', sep='\t', header=None)
similar_train_df.columns = ['qid', 'similar', 'random']
similar_train_dataset = Dataset.from_pandas(similar_train_df)
qid_dict = {value: index for index, value in enumerate(dataset['qid'])}
ds = DatasetDict({
    'train': similar_train_dataset.select(range(100)),
    'validation': similar_train_dataset.select(range(101, 120)),
})


## defining tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

context_length = 512
eos_id = tokenizer.eos_token_id#tokenizer(tokenizer.eos_token)['input_ids'][0]
istring = np.vectorize(lambda x: isinstance(x, str))
#ssplit_similars = np.vectorize(lambda x: [dataset['body'][qid_dict[int(num)]] for num in x.split()])
ssplit_base = np.vectorize(lambda x: np.array(dataset['body'])[np.array(dataset['qid']) == x])
def tokenize(element):
    print('I enter tokenize')
    #count = 0
    #for text in element['body']:
    #    if isinstance(text, str) == False:
    #        print('THIS INPUT IS NOT STRING!!!')
    #        print('text is ', text, ' element body is ', element['body'][count -1: count +1])
    #    count += 1
    similar_questions_str = [[dataset['body'][qid_dict[int(num)]] for num in x.split()] for x in element['similar']] ## this is a numpy array!
    print('I pass similar_questions_str')
    #base_question_str = ssplit_base(np.array(element['qid']))
    base_question_str = [dataset['body'][qid_dict[x]]]
    istring_check = istring(base_question_str)
    similar_questions_str = list(similar_questions_str[istring_check])
    base_question_str = list(base_question_str[istring_check])
    print('how about here?!')
    outputs_base = tokenizer(
        base_question_str,
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    outputs_similar = {'input_ids': [tokenizer(
        list_of_similars,
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )['input_ids'].append(eos_id) for list_of_similars in similar_questions_str]}

    input_batch = []
    similars_batch = []
    print('Im here!!')
    for length, input_ids_base, list_of_similars_ids in zip(outputs_base['length'], outputs_base['input_ids'], outputs_similar['input_ids']):
        #print('length is ', length)
        if length <= context_length:
            input_ids_base.append(eos_id)
            #while len(input_ids) < context_length:
            #    input_ids.append(eos_id)
            input_batch.append(input_ids_base)
            similars_batch.append(list_of_similars_ids)
    #print('batch length is ', len(input_batch))
    return {'base': input_batch, 'disimilar': similars_batch}

#tokenizing the dataset
tokenized_datasets = ds.map(
    tokenize, batched=True, remove_columns=ds["train"].column_names
)
tokenized_datasets.save_to_disk('./model3-outputs-gpt2tokenizer-askubuntubody-margintraining-150000train-10000test')