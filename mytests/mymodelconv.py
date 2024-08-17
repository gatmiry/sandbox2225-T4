from transformers import PretrainedModel

class MyModel(PretrainedModel):
    def __init__(self, config, conv_length):
        super.__init__(config)
        self.conv_length = conv_length
    