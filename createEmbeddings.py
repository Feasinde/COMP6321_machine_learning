import gensim
import xml.etree.ElementTree as ET

tree = ET.parse("dataset.xml")
root = tree.getroot()

documents = []
for convo in root:
    sentences = []
    for message in convo:
        sentences.append(message[2].text)
    sentences = [sentence for sentence in sentences if type(sentence)==str]
    sentences = [sentence.replace('&amp;','&').replace('&apos;',"'").lower() for sentence in sentences]
    documents.append(sentences)
documents = [" ".join(document) for document in documents]
documents = [gensim.utils.simple_preprocess(document) for document in documents]
model = gensim.models.Word2Vec(documents,min_count=1)
print(model.wv.most_similar("girl"))
print(model.wv.most_similar("panties"))
print(model.wv.most_similar("table"))
print(model.wv.most_similar("wet"))
print(model.wv.most_similar("perv")) # LMAO 'canadian'
