# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="nlpodyssey/bert-italian-uncased-iptc-headlines")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nlpodyssey/bert-italian-uncased-iptc-headlines")
model = AutoModelForSequenceClassification.from_pretrained("nlpodyssey/bert-italian-uncased-iptc-headlines")

################################################################################################


#pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")