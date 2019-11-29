import torch,gensim
from dataset import SexPredDataset
from sexpred import SexPred
from torch.utils.data import DataLoader
from tqdm import tqdm

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

emb=gensim.models.KeyedVectors.load_word2vec_format("embeddings.bin",binary=True)
emb_tensor=torch.FloatTensor(emb.vectors)
ds=SexPredDataset("dataset.xml","pervs.txt")
ds=DataLoader(ds,batch_size=5)
for i in ds:
    l=i
    print(l[1].shape)
    exit(0)
# train_it = tqdm(ds,desc='Training',leave=False)
# v=ds[1024][0]
# p=ds[1024][1]
# v=torch.unsqueeze(v,0).to(device)
# p=torch.unsqueeze(p,0).to(device)
# t=v,p
# sp=SexPred(emb_tensor).to(device)

# print(sp(t))
