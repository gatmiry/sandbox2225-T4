import torch
import numpy as np
import requests
context_length = 512

class EmbedCalculator:
    def calculate_embedding(self, inputs):
        url = 'https://blog.yintat.com/dqjoBCx0P2k7/api/models/unified-completion'
        header =  {'Authorization': ''}
        data = {'model': 'text-embedding-3-large', 'input': inputs}
        response = requests.post(url, headers=header, json=data, timeout=600)
        import json
        data = json.loads(response.text)
        return torch.tensor([element['embedding'] for element in data['data']])



   
