import torch
from torch import nn
import torch.nn.functional as F
from src.dataset import tensorFromSentence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self,
                 embeddings_tensor,
                 hidden_size=256,
                 dropout=0.5):
        super(Encoder, self).__init__()
        self.embedding_size=len(embeddings_tensor[0])
        self.hidden_size=hidden_size
        self.dropout_p=dropout
        self.embedding=nn.Embedding.from_pretrained(embeddings_tensor)
        self.gru=nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False)
        
    def forward(self,input_tensor,hidden):
        output=self.embedding(input_tensor)
        output,hidden=self.gru(output,hidden)
        return output,hidden
    
    def initHidden(self,batch_size):
        n=1 if self.gru.bidirectional==False else 2
        l=self.gru.num_layers
        return torch.zeros(n*l, batch_size, self.hidden_size)

class Decoder(nn.Module):
    def __init__(self,
                 embeddings_tensor,
                 hidden_size=256,
                 dropout=0.5):
        super(Decoder, self).__init__()
        self.embedding_size=len(embeddings_tensor[0])
        self.hidden_size=hidden_size
        self.dropout_p=dropout
        
        self.embedding=nn.Embedding.from_pretrained(embeddings_tensor)
        self.gru=nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False)
        n=1 if self.gru.bidirectional==False else 2
        self.out=nn.Linear(self.hidden_size*n, len(embeddings_tensor))

    def forward(self,input_tensor,hidden):
        output=self.embedding(input_tensor)
        output=F.relu(output)
        output,hidden=self.gru(output,hidden)
        output=self.out(output)
        return output,hidden

class SexPred(nn.Module):
    """
    Encoder-decoder model that imitates the behaviour of an online
    sexual predator
    """
    def __init__(self,embeddings_tensor,hidden_size=256):
        super(SexPred,self).__init__()
        self.embeddings_tensor=embeddings_tensor
        self.hidden_size=hidden_size
        self.encoder=Encoder(self.embeddings_tensor,hidden_size=self.hidden_size)
        self.decoder=Decoder(self.embeddings_tensor,hidden_size=self.hidden_size)
    
    def forward(self,vic_perv):
        v=vic_perv[0]
        p=vic_perv[1]
        batch_size=len(v)
        h_enc=self.encoder.initHidden(batch_size).to(device)
        _,h=self.encoder(v,h_enc)
        o,_=self.decoder(p,h)
        return o
    
#     def predict(self,input_tensor,hidden):
#         output=self.decoder(input_tensor,hidden)
        
        
        
