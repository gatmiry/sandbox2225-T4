import torch.nn as nn
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2SdpaAttention
from transformers.models.gpt2.modeling_gpt2 import Conv1D
class MyGPT2SdpaAttention(GPT2SdpaAttention):
    def __init__(self, *args, **kwargs):
        config = kwargs.get('config', None)
        super().__init__(*args, **kwargs)
        self.num_embeddings = 1
        self.my_c_attn_words = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.my_c_attn_embeddings = Conv1D(3 * self.embed_dim, self.embed_dim)
        #self.c_attns = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def set_num_embeddings(self, num_embeddings):
        self.num_embeddings = num_embeddings

    def forward(self, hidden_states, *args, **kwargs):
        #print('args is ', args)
        #print('kwargs is ', kwargs)
        #outputs = super().forward(*args, **kwargs)
        layer_past = kwargs.get('layer_past', None)
        use_cache = kwargs.get('use_cache', None)
        output_attentions = kwargs.get('output_attentions', None)
        #hidden_states = kwargs.get('hidden_states', None)
        attention_mask = kwargs.get('attention_mask', None)
        head_mask = kwargs.get('head_mask', None)
        encoder_hidden_states = kwargs.get('encoder_hidden_states', None)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            print('there is cross attention!')
        #print('hidden state is ', hidden_states)
        ## here we are assuming that this module is never instantiated to implement cross attention, here is the part that we change compared to GPT2Attention
        hidden_embeddings = hidden_states[:,:self.num_embeddings,:]
        hidden_words = hidden_states[:,self.num_embeddings:,:]
        output_embeddings = self.my_c_attn_embeddings(hidden_embeddings)
        output_words = self.my_c_attn_words(hidden_words)
        output_states = torch.cat((output_embeddings, output_words), dim=1)
        query, key, value = output_states.split(self.split_size, dim=2)
        #print('query shape is ', query.shape)
        
        ## below is almost copy paste from GPT2Attention class!
        #query_embeddings = query[:,:self.num_embeddings,:]
        #query_words = query[:,self.num_embeddings:,:]
        #query_embeddings = self._split_heads(query_embeddings, self.num_heads, self.head_dim)
        #query_words = self._split_heads(query_words, self.num_heads, self.head_dim)
        #print('query_words shape after split is ', query_words.shape, ' query_embeddings shape is ', query_embeddings.shape)
        ##key_embeddings = key[:,:self.num_embeddings,:]
        #key_words = key[:,self.num_embeddings:,:]
        #key_embeddings = self._split_heads(key_embeddings, self.num_heads, self.head_dim)
        #key_words = self._split_heads(key_words, self.num_heads, self.head_dim)
        #value_embeddings = value[:,:self.num_embeddings,:]
        #value_words = value[:,self.num_embeddings:,:]
        #value_embeddings = self._split_heads(value_embeddings, self.num_heads, self.head_dim)
        #value_words = self._split_heads(value_words, self.num_heads, self.head_dim)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        ##
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
        #return outputs