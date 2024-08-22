def flip(state):
    if state == 'User':
        return 'Assistant'
    if state == 'Assistant':
        return 'User'
    


with open('../conversational-dataset/conversations_gpt35_1.txt', 'r') as file:
    conversation = []
    conversations = []
    paragraph = ''
    state = 'User'
    new_conversation = 'True'
    count = 0
    #max_len = 0
    for index, line in enumerate(file):
        #count += 1
        line = line.strip()
        if len(line) < 5:
            continue
        if line[:10] == 'Background':
            conversation = []
            paragraph = line
            state = 'User'
        if line[:4] == state or line[:9] == state:
            #print('im here! state is ', state)
            if paragraph != '':
                #if len(paragraph) > max_len:
                #    max_len = len(paragraph)
                conversation.append(paragraph)
                count += 1
            paragraph = line
            state = flip(state)
        else:
            paragraph = paragraph + ' \n ' + line
        if line[-13:] == '<|endoftext|>':
            conversation.append(paragraph)
            if count < 50:
                conversations.append({'conversation': conversation})
            #print('count is ', count)
            conversation = []
            count = 0
            paragraph = ''
            state = 'User'
            if len(conversations) > 310000:
                break

###
train_data = conversations[:300000]
test_data = conversations[300001:310000]

###
from datasets import Dataset, DatasetDict
train_data = Dataset.from_list(train_data)
test_data = Dataset.from_list(test_data)
dataset = DatasetDict({'train': train_data, 'validation': test_data})


###
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
max_length = 2048
max_conversation_length = 50
def tokenize(element):
    #print('element is ', element)
    outputs = [tokenizer(conversation)['input_ids'] for conversation in element['conversation']]
    conversations_batch = []
    for conversation in outputs:
        flag = True
        if len(conversation) > max_conversation_length:
            flag = False
        for paragraph in conversation:
            paragraph.append(tokenizer.eos_token_id)
            if len(paragraph) > max_length:
                flag = False
        if flag:
            conversations_batch.append(conversation)
    return {'input_ids': conversations_batch}


##
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset['train'].column_names)

## 
tokenized_dataset.save_to_disk('./conversational-dataset-300000')