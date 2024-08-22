from transformers import DataCollatorWithPadding

class DataCollatorForTextSimilarity(DataCollatorWithPadding):
    #sub_batch_size = 2
    def __init__(self, tokenizer, sub_batch_size):
        self.sub_batch_size = sub_batch_size
        self.tokenizer = tokenizer
    def __call__(self, features):
        #print('i have been called!!!')
        text_batch = []
        count = 0
        indices = []
        sign = []
        for bag in features:
            #print('bag keys are ', bag.keys())
            text_batch.append(bag['input_ids'])
            text_batch = [*text_batch, *bag['similar'], *bag['random']]
            indices = [*indices, *([count]*(1 + len(bag['similar']) + len(bag['random'])))]
            sign.append('base')
            sign = [*sign, *(['similar']*(len(bag['similar']))), *(['random']*(len(bag['random'])))]
            count += 1
        list_of_sub_batches = []
        sub_batch_num = int(len(sign) / self.sub_batch_size)
        if sub_batch_num * self.sub_batch_size < len(sign):
            sub_batch_num += 1
        for i in range(sub_batch_num):
            list_of_sub_batches.append(self.tokenizer.pad({'input_ids': text_batch[i*self.sub_batch_size: (i+1)*self.sub_batch_size]}, padding=self.padding, return_tensors='pt')['input_ids'])
        return {'list_of_sub_batches': list_of_sub_batches, 'sign': sign, 'indices': indices}
    
class DataCollatorForConversationTraining(DataCollatorWithPadding):
    def __init__(self, tokenizer, max_num_paragraphs):
        self.max_num_paragraphs = max_num_paragraphs
        self.tokenizer = tokenizer
    def __call__(self, features):
        conv_len = self.max_num_paragraphs
        for short_conversation_list in features:
            if len(short_conversation_list['input_ids']) < conv_len:
                conv_len = len(short_conversation_list['input_ids'])
                #print('conv len is ', conv_len)
        conversations_over_time = [[] for i in range(conv_len)]
        for short_conversation_list in features:
            for i in range(conv_len):
                #print('short_conversation_list is ', short_conversation_list)
                conversations_over_time[i].append(short_conversation_list['input_ids'][i])
        conversations_over_time = [self.tokenizer.pad({'input_ids': conversations}, padding=self.padding, return_tensors='pt')['input_ids'] for conversations in conversations_over_time]
        attention_masks_over_time = [self.tokenizer.pad({'input_ids': conversations}, padding=self.padding, return_tensors='pt')['attention_mask'] for conversations in conversations_over_time]
        return {'input_ids_list': conversations_over_time, 'labels_list': conversations_over_time, 'attention_mask_list': attention_masks_over_time}