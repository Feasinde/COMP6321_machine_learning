import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cpu")

class Encoder(nn.Module):
    def __init__(self,
                 embeddings_tensor,
                 hidden_size=40,
                 dropout=0.5):
        super(Encoder, self).__init__()
        self.__embedding_size=len(embeddings_tensor[0])
        self.__hidden_size=hidden_size
        self.__dropout_p=dropout
        
        self.__embedding=nn.Embedding.from_pretrained(embeddings_tensor)
        self.__dropout=nn.Dropout(self.__dropout_p)
        self.__gru=nn.GRU(
            input_size=self.__embedding_size,
            hidden_size=self.__hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=self.__dropout_p,
            bidirectional=False)
        
    def forward(self,input_tensor,hidden):
        output=self.__embedding(input_tensor)
#         output=self.__dropout(output)
        output,hidden=self.__gru(output,hidden)
        return output,hidden
    
    def initHidden(self):
        n=1 if self.__gru.bidirectional==False else 2
        l=self.__gru.num_layers
        return torch.zeros(n*l, 1, self.__hidden_size, device=device)

class Decoder(nn.Module):
    def __init__(self,
                 embeddings_tensor,
                 hidden_size=40,
                 dropout=0.5):
        super(Decoder, self).__init__()
        self.__embedding_size=len(embeddings_tensor[0])
        self.__hidden_size=hidden_size
        self.__dropout_p=dropout
        
        self.__embedding=nn.Embedding.from_pretrained(embeddings_tensor)
        self.__gru=nn.GRU(
            input_size=self.__embedding_size,
            hidden_size=self.__hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=self.__dropout_p,
            bidirectional=False)
        n=1 if self.__gru.bidirectional==False else 2
        self.__out=nn.Linear(self.__hidden_size*n, len(embeddings_tensor))

    def forward(self,input_tensor,hidden):
        output=self.__embedding(input_tensor)
        output=F.relu(output)
        output,hidden=self.__gru(output,hidden)
        output=self.__out(output)
        return output,hidden

class SexPred(nn.Module):
    def __init__(self,embeddings_tensor):
        super(SexPred,self).__init__()
        self.__embeddings_tensor=embeddings_tensor
        self.__encoder=Encoder(self.__embeddings_tensor)
        self.__decoder=Decoder(self.__embeddings_tensor)
    
    def forward(self,vic_perv):
        v=vic_perv[0]
        p=vic_perv[1]
        h_enc=self.__encoder.initHidden()
        _,h=self.__encoder(v,h_enc)
        o,_=self.__decoder(p,h)
        return o