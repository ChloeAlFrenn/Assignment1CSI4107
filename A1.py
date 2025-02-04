import json
import re
import math
from collections import defaultdict
from nltk.stem import PorterStemmer

def load_stopwords(file_path):
    """
    Loads stopwords from a file into a set
    """
    with open(file_path, "r") as file:
        return set(word.strip() for word in file)

def preprocess_text(text, stop_words, stem=False):
    """
    Preprocess a text: removes unwanted characters, tokenize, remove stopwords, and can apply stemming if stem
    is set to True
    """
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
        for line in file:
            document = json.loads(line.strip())
            doc_id = document["_id"]
            text = document.get("text", "")
            tokens = preprocess_text(text, stop_words, stem)
            for token in tokens:
                inverted_index[token][doc_id] += 1
    return inverted_index

 
def compute_tf_idf(inverted_index, total_docs):
    """
    Computes TF-IDF weights for all terms in the inverted index and document L2 norms.
    
    Formula:
    TF-IDF = (Term Frequency in Document) * log(Total Documents / Document Frequency)
    L2 Norm = sqrt(Σ(TF-IDF_weight^2) for all terms in document)
    
    Args:
        inverted_index: Dict[term, Dict[doc_id, term_frequency]]
        total_docs: Total number of documents in the collection
    
    Returns:
        tf_idf_index: Dict[term, Dict[doc_id, tf_idf_weight]]
        doc_norms: Dict[doc_id, L2_norm_of_document_vector]
    """

    tf_idf_index = {}
    doc_norms = defaultdict(float)

    
    for term, doc_freqs in inverted_index.items():
        # Document Frequency (DF) = number of documents containing the term
        df = len(doc_freqs)

        # Inverse Document Frequency (IDF) = log(total_docs / df)
        idf = math.log(total_docs / df)

        tf_idf_index[term] = {}

        # For each document containing this term
        for doc_id, tf in doc_freqs.items():
            # TF-IDF Calculation:
            # Term Frequency (TF) = raw count in document
            # TF-IDF = TF * IDF           
            tf_idf = tf * idf

            # Store TF-IDF weight
            tf_idf_index[term][doc_id] = tf_idf

            # Accumulate squared TF-IDF values for L2 norm calculation
            doc_norms[doc_id] += tf_idf ** 2   # ||document_vector||^2
    
    # Compute L2 norms by taking square root of sum of squares
    doc_norms = {doc_id: math.sqrt(score) for doc_id, score in doc_norms.items()}

    return tf_idf_index, doc_norms



def compute_cosine_similarity(query_tokens, tf_idf_index, total_docs, doc_norms):
    """
    Computes cosine similarity between query and all documents.
    
    Formula:
    Cosine Similarity = (Q · D) / (||Q|| * ||D||)
    Where:
    - Q · D = Σ(q_i * d_i) for all terms
    - ||Q|| = sqrt(Σ(q_i^2))
    - ||D|| = precomputed document norm
    
    Args:
        query_tokens: List of terms in the query
        tf_idf_index: Precomputed TF-IDF weights from compute_tf_idf()
        total_docs: Total number of documents
        doc_norms: Precomputed document norms from compute_tf_idf()
    
    Returns:
        Sorted list of (doc_id, score) pairs in descending order
    """
    scores = defaultdict(float)
    query_vector = defaultdict(float)

    # Query Processing --------------------------------------------------------
    # Step 1: Build query vector (TF-IDF representation)
    query_terms_count = defaultdict(int)
    
    # Calculate raw term frequencies in query
    for term in query_tokens:
        query_terms_count[term] += 1
    
    query_norm = 0.0
    # Convert raw counts to TF-IDF weights
    for term, tf in query_terms_count.items():
        if term in tf_idf_index:
            # Get document frequency (DF) for the term
            df = len(tf_idf_index[term])

             # Calculate IDF using same formula as documents
            idf = math.log(total_docs / df)

            # TF-IDF = TF * IDF
            tf_idf = tf * idf

            # Store TF-IDF weight
            query_vector[term] = tf_idf

            # Accumulate squared values for query norm calculation
            query_norm += tf_idf ** 2

    # Complete query norm calculation        
    query_norm = math.sqrt(query_norm)

    # Compute dot product and normalize
    for term, q_weight in query_vector.items():
        if term in tf_idf_index:
            for doc_id, d_weight in tf_idf_index[term].items():
                scores[doc_id] += q_weight * d_weight # Q · D = Σ(q_i * d_i) for all terms

    # Normalize scores by query and document norms
    for doc_id in scores:
        # Handle edge cases where norm is zero (avoid division by zero)
        if doc_norms.get(doc_id, 0) == 0 or query_norm == 0:
            scores[doc_id] = 0.0
        else:
            scores[doc_id] /= (query_norm * doc_norms[doc_id])
            
    # Return documents sorted by descending similarity score        
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)




# Main function to load files and run the process
def main(corpus_file, stopwords_file, queries_file, output_file, stem=False):
    # Load stopwords from file
    stop_words = load_stopwords(stopwords_file)
    
    # Build the inverted index from the corpus (now uses document["_id"])
    inverted_index = create_inverted_index(corpus_file, stop_words, stem)
    
    # Count total number of documents by reading the corpus file
    with open(corpus_file, "r") as file:
        total_docs = sum(1 for _ in file)
    
    # Compute TF-IDF and document norms
    tf_idf_index, doc_norms = compute_tf_idf(inverted_index, total_docs)  # Now returns doc_norms
    
    # Process queries and write results
    with open(queries_file, "r") as file, open(output_file, "w") as out_file:
        for line in file:
            query_data = json.loads(line.strip())
            query_id = query_data["_id"]
            query_text = query_data["text"]
            
            # Preprocess query tokens
            query_tokens = preprocess_text(query_text, stop_words, stem)
            
            # Compute cosine similarity
            ranked_results = compute_cosine_similarity(
                query_tokens, 
                tf_idf_index, 
                total_docs, 
                doc_norms 
            )
            
            for rank, (doc_id, score) in enumerate(ranked_results[:100], start=1):
                out_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} Assignment1\n")
    


corpus_file = "scifact/corpus.jsonl" 
queries_file = "scifact/queries.jsonl" 
stopwords_file = "stopwords.txt" 
output_file = "Results.txt"     

main(corpus_file, stopwords_file, queries_file, output_file, stem=False)