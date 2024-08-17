## loading askubuntu
import pandas as pd
import numpy as np
from datasets import load_from_disk
df = pd.read_csv("texts_raw_fixed.txt", sep="\t", header=None)
df.columns = ['qid', 'title', 'body']
from datasets import Dataset, DatasetDict
dataset = Dataset.from_pandas(df)
dataset_ids = load_from_disk('./model3-outputs-gpt2tokenizer-askubuntubody-150000train-10000test')
similar_train_df = pd.read_csv('train_random.txt', sep='\t', header=None)
similar_train_df.columns = ['qid', 'similar', 'random']
similar_train_dataset = Dataset.from_pandas(similar_train_df)
qid_dict = {value: index for index, value in enumerate(dataset['qid'])}
ds = DatasetDict({
    'train': similar_train_dataset.select(range(10000)),
    'validation': similar_train_dataset.select(range(10001, 12000)),
})

## defining mylist
mylist = dataset_ids['train']['input_ids']

## defining tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

context_length = 512
eos_id = tokenizer.eos_token_id#tokenizer(tokenizer.eos_token)['input_ids'][0]
istring = np.vectorize(lambda x: isinstance(x, str))
#ssplit_similars = np.vectorize(lambda x: [dataset['body'][qid_dict[int(num)]] for num in x.split()])
#ssplit_base = np.vectorize(lambda x: np.array(dataset['body'])[np.array(dataset['qid']) == x])
def tokenize(element):
    base_questions = [mylist[qid_dict[qid]] for qid in element['qid']]
    similar_questions = [[mylist[qid_dict[int(num)]] for num in x.split()] 
                             for x in element['similar']]
    random_questions = [[mylist[qid_dict[int(num)]] for num in x.split()] 
                             for x in element['random']]
    
   
    #similars_batch = []
    for base_question_ids in base_questions:
        base_question_ids.append(eos_id)
    for list_of_similars in similar_questions:
        for similar_ids in list_of_similars:
            similar_ids.append(eos_id)
    for list_of_randoms in random_questions:
        for random_ids in list_of_randoms:
            random_ids.append(eos_id)
        #similars_batch.append(list_of_similars)
        #random_batch.append(list_of_random)
        #print('input_batch is ', input_batch)
    return {'input_ids': base_questions, 'similar': similar_questions, 'random': random_questions}

#tokenizing the dataset
tokenized_datasets = ds.map(
    tokenize, batched=True, remove_columns=ds["train"].column_names, batch_size=10
)
#similar_questions =  

#mylist = dataset_ids['train']['input_ids']
#for x in ds['train']['similar']:
#    print('Im here')
#    print('x.split is ', x.split())
#    print([dataset_ids['train']['input_ids'][qid_dict[int(num)]] for num in x.split()])
#    print('Im done')

tokenized_datasets.save_to_disk('./model3-outputs-gpt2tokenizer-askubuntubody-margintraining-150000train-10000test')