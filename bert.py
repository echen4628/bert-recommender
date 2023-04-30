import torch
from transformers import BertTokenizer, BertModel
import pdb
import numpy as np
from tqdm import tqdm
class BertForEmbedding():
    def __init__(self, config='bert-base-uncased', device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.bert = BertModel.from_pretrained(config).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(config)
        self.device = device
    def clip(self, tokens):
        return tokens[:511] + [102]
    @torch.no_grad()
    def bertify_single_abstract(self, abstract):
        # tokenize inputs and add [CLS]
        tokens = self.tokenizer.encode(abstract, add_special_tokens=True)
        if len(tokens) > 512:
            tokens = self.clip(tokens)
        # pass to model
        input_tokens = torch.tensor([tokens]).to(self.device)
        output = self.bert(input_tokens) #torch.Size([batch, token_length, 768])

        # return only the embeddings corresponding [CLS]
        # return output.last_hidden_state[:,0,:]

        # return average of the hidden states
        return torch.mean(output.last_hidden_state, axis=1)
        pdb.set_trace()
    
    @torch.no_grad()
    def bertify_abstracts(self, abstract_l):
        # tokenize inputs and add [CLS] 
        outputs = []
        for idx, abstract in tqdm(enumerate(abstract_l)):
            try:
                output = self.bertify_single_abstract(abstract)
            except:
                pdb.set_trace()
            outputs.append(output[0].cpu().numpy())
        outputs = np.array(outputs)
        return outputs
    



