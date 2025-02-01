import json
import re
from collections import defaultdict
from nltk.stem import PorterStemmer

# Load stopwords from a file into a set
def load_stopwords(file_path):
    with open(file_path, "r") as file:
        return set(word.strip() for word in file)

# Preprocess a text: removes unwanted characters, tokenize, remove stopwords, and (optionally) apply stemming
def preprocess_text(text, stop_words, stem=False):
    # Remove all characters that aren't letters or spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = text.lower().split()
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Optional: Apply stemming
    if stem:
        stemmer = PorterStemmer()
        filtered_tokens = [stemmer.stem(word) for word in filtered_tokens]

    return filtered_tokens

# Build the inverted index
def create_inverted_index(corpus_file, stop_words, stem=False):
    
    inverted_index = defaultdict(lambda: defaultdict(int))  # term -> {doc_id: tf}
    
    with open(corpus_file, "r") as file:
        for doc_num, line in enumerate(file):
            document = json.loads(line.strip())  
            text = document.get("text", "")  
            
            # Preprocess the text
            tokens = preprocess_text(text, stop_words, stem)
            
            # Update the inverted index
            for token in tokens:
                inverted_index[token][doc_num] += 1  # Increment term frequency for this document

    return inverted_index

# Main function to load files and run the process
def main(corpus_file, stopwords_file, stem=False):
    # Load stopwords from file
    stop_words = load_stopwords(stopwords_file)
    
    # Build the inverted index from the corpus
    inverted_index = create_inverted_index(corpus_file, stop_words, stem)
    
    # Output the inverted index (just the first few entries to check)
    print("Sample inverted index:")
    for term, doc_freq in list(inverted_index.items())[:10]:  # Print only first 10 terms
        print(f"Term: {term}, Documents: {dict(doc_freq)}")


corpus_file = "scifact/corpus.jsonl"  
stopwords_file = "stopwords.txt"      

# Run the preprocessing and indexing steps
main(corpus_file, stopwords_file, stem=False)
