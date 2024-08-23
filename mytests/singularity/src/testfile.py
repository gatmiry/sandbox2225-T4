log_history_file = '/mnt/my_output/myfile.json'
import json
with open(log_history_file, 'w') as f:
    json.dump('Hello!', f, indent=4)

import torch
print('torch version is ', torch.__version__)
from transformers import AutoModel
print('successfully imported transformer library!')

mymodel = AutoModel.from_pretrained('gpt2')
print('model is ', mymodel)