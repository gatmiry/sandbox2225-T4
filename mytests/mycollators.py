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