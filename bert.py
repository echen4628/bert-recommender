import torch
from transformers import BertTokenizer, BertModel

class BertForEmbedding():
    def __init__(self, config='bert-base-uncased'):
        self.bert = BertModel.from_pretrained(config)
        self.tokenizer = BertTokenizer.from_pretrained(config)
        
    @torch.no_grad()
    def bertify_single_abstract(self, abstract):
        # tokenize inputs and add [CLS]
        tokens = self.tokenizer.encode(abstract, add_special_tokens=True)

        # pass to model
        input_tokens = torch.tensor([tokens])
        output = self.bert(input_tokens) #torch.Size([batch, token_length, 768])

        # return only the embeddings corresponding [CLS]
        return output.last_hidden_state[:,0,:] 


