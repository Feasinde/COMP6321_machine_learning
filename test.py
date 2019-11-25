import torch,gensim
from dataset import SexPredDataset
from sexpred import SexPred

device=torch.device("cpu")

emb=gensim.models.KeyedVectors.load_word2vec_format("embeddings.bin",binary=True)
emb_tensor=torch.FloatTensor(emb.vectors)
ds=SexPredDataset("dataset.xml","pervs.txt")
v=ds[1024][0]
p=ds[1024][1]
v=torch.unsqueeze(v,0).to(device)
p=torch.unsqueeze(p,0).to(device)
t=v,p
sp=SexPred(emb_tensor).to(device)
print(sp(t))
