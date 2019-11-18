import torch
import xml.etree.ElementTree as ET
import torch.nn.functional as F
from torch.utils.data import Dataset


def mergeSameAuthors(conv):
    """Merges contiguous messages by the same author
    
    In: list of id-message tuples where several contiguous messages might belong to the same author
    Out: list of id-message tuples alternating authors
    
    """
    conversation_=conv.copy()
    change=True
    while change:
        change=False
        for index in range(len(conversation_)-1):
            if conversation_[index][0]==conversation_[index+1][0]:
                try:
                    conversation_[index][1]+=" "+conversation_[index+1][1]
                except TypeError:
                    pass
                conversation_[index+1][0]=''
                change=True
        conversation_ = [message for message in conversation_ if not message[0]=='']
    return conversation_    

def pervVictimExchange(conv,pervs):
    """Converts a conversation into a list of perv-victim dialogue tuples
    
    In: List of author-message tuples and list of pervs
    Out: List of perv-victim exchange tuples
    
    """
    perv_victim=[]
    n=0
    if conv[0][0] in pervs:
        n=1
    for i in range(n,len(conv)-1,2):
        perv_victim.append([conv[i][1],conv[i+1][1]])
    return perv_victim

def tensorFromSentence(embeddings,sentence,):
    """Returns a PyTorch tensor object from a string
    
    In: embeddings model and a sentence string
    Out: long tensor object
    
    """
    index_list = []
    sentence = sentence.split(' ')
    word2index = {token: token_index for token_index, token in enumerate(embeddings.index2word)}
    for word in sentence:
        try:
            index = word2index[word]
        except:
            index = 0
        index_list.append(index)
    return torch.tensor(index_list, dtype=torch.long)

class SexPredDataset(Dataset):
    """Dataset of predator messages"""
    def __init__(self,xml_file,txt_file,max_length=40):
        """
        
        Args:
        xml_file (string): path to xml file of dataset
        txt_file (string): path to txt file containing the list of predators
        max_length (int): max number of tokens in each exchange
        
        """
        self.__max_length = max_length
        self.__root = ET.parse(xml_file).getroot()
        self.__perv_convos = []
        with open(txt_file) as file:
            pervs = file.readlines()
        pervs = [perv.strip() for perv in pervs]
        for convo in self.__root:
            author1=convo[0][0].text
            n=1
            try:
                while convo[n][0].text==author1: n+=1
            except IndexError: 
                continue
            author2=convo[n][0].text
            if author1 in pervs or author2 in pervs: self.__perv_convos.append(convo)
        self.__perv_convos = [[[message[0].text, message[2].text] for message in conv] for conv in self.__perv_convos]
        self.__perv_convos = [mergeSameAuthors(conv) for conv in self.__perv_convos]
        self.__perv_victim = []
        for convo in self.__perv_convos:
            self.__perv_victim+=pervVictimExchange(convo,pervs)
                    
    def __len__(self): return len(self.__perv_victim)
    
    def __getitem__(self,index):
        input_tensor = tensorFromSentence(embeddings,self.__perv_victim[index][0])
        output_tensor = tensorFromSentence(embeddings,self.__perv_victim[index][1])
        victim_padding = self.__max_length - len(input_tensor)
        perv_padding = self.__max_length - len(output_tensor)
        input_tensor,output_tensor = F.pad(input_tensor, pad=(0, victim_padding), mode='constant', value=0), F.pad(output_tensor, pad=(0, perv_padding), mode='constant', value=0)
        return input_tensor,output_tensor