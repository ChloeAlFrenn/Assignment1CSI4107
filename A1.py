import json
import re
import math
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

# Compute the TF-IDF 
def compute_tf_idf(inverted_index, total_docs):
    tf_idf_index = {}
    
    for term, doc_freqs in inverted_index.items():
        df = len(doc_freqs)  # Number of documents containing the term
        idf = math.log(total_docs / (df))  # Compute IDF 
        
        tf_idf_index[term] = {}
        for doc_id, tf in doc_freqs.items():
            tf_idf_index[term][doc_id] = tf * idf  # Compute TF-IDF
    
    return tf_idf_index


# Compute the cosine similarity scores between a query and each document
def compute_cosine_similarity(query_tokens, tf_idf_index, total_docs):
    scores = defaultdict(float)
    query_vector = defaultdict(float)
    
    # Compute query TF-IDF vector
    for term in query_tokens:
        if term in tf_idf_index:
            df = len(tf_idf_index[term])
            idf = math.log(total_docs / (1 + df))
            query_vector[term] += idf
    
    # Compute document similarity scores
    for term, weight in query_vector.items():
        if term in tf_idf_index:
            for doc_id, doc_weight in tf_idf_index[term].items():
                scores[doc_id] += weight * doc_weight
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)  # Sort by score descending




# Main function to load files and run the process
def main(corpus_file, stopwords_file, queries_file, output_file, stem=False):
    # Count total number of documents
    with open(corpus_file, "r") as file:
        total_docs = sum(1 for _ in file)  

    # Load stopwords from file
    stop_words = load_stopwords(stopwords_file)
    
    # Build the inverted index from the corpus
    inverted_index = create_inverted_index(corpus_file, stop_words, stem)

    # Compute the TF-IDF 
    tf_idf_index = compute_tf_idf(inverted_index, total_docs)
    
    # Open the queries and calculate their cosin scores and write the result file
    with open(queries_file, "r") as file, open(output_file, "w") as out_file:
        for line in file:
            query_data = json.loads(line.strip())
            query_id = query_data["_id"]
            query_text = query_data["text"]
            
            # Preprocess query
            query_tokens = preprocess_text(query_text, stop_words, stem)
            
            # Compute cosine similarity
            ranked_results = compute_cosine_similarity(query_tokens, tf_idf_index, total_docs)
            
            # Write results to file
            for rank, (doc_id, score) in enumerate(ranked_results[:100], start=1):
                out_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} Assignment1\n")
    


corpus_file = "scifact/corpus.jsonl" 
queries_file = "scifact/queries.jsonl" 
stopwords_file = "stopwords.txt" 
output_file = "Results.txt"     

# Run the preprocessing and indexing steps
main(corpus_file, stopwords_file, queries_file, output_file, stem=False)
