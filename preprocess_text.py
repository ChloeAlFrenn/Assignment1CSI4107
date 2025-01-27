import json
import re
from nltk.stem import PorterStemmer #add the import to RM

# Load stopwords from a file into a set. 
def load_stopwords(file_path):
    with open(file_path, "r") as file:
        return set(word.strip() for word in file)

# Preprocesse a text: removes unwanted characters, tokenize, remove stopwords, and (optionnal) apply stemming. 
def preprocess_text(text, stop_words, stem=False):
    #removes all characters that aren't letters or spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = text.lower().split()
    
    filtered_tokens = []
    for word in tokens:
        if word not in stop_words:
            filtered_tokens.append(word)
    tokens = filtered_tokens

    #optional: stemming
    if stem:
        stemmer = PorterStemmer()
        stemmed_tokens = []
        for word in filtered_tokens:
            stemmed_tokens.append(stemmer.stem(word))
        tokens = stemmed_tokens

    
    return tokens

# Process a file: loads the file and receive the preprocessed tokens.
def process_file(text_file, stop_words, stem=False):
    with open(text_file, "r") as file:
        for line in file:
            document = json.loads(line.strip())  
            text = document.get("text", "")  
            
            processed_tokens = preprocess_text(text, stop_words, stem)
            
            #printing for now change later
            print(processed_tokens)  



stopwords_file = "stopwords.txt" 
corpus_file = "scifact/corpus.jsonl"  
queries_file="scifact/queries.jsonl"
    
stop_words = load_stopwords(stopwords_file)
    
#pass stem as true to apply stemming)
process_file(corpus_file, stop_words, stem=False)
process_file(queries_file,stop_words, stem=False)
