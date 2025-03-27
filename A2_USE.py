#pip install tensorflow tensorflow-hub numpy scikit-learn
#Run A1 before
#run /Applications/Python\ 3.11/Install\ Certificates.command if you get certificate errors

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub

# Load Universal Sentence Encoder model
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def compute_use_embeddings(texts):
    """Generates USE embeddings for a list of texts."""
    return use_model(texts).numpy()

def rerank_documents(query_text, initial_results, doc_texts):
    """Re-ranks the top 100 documents using USE embeddings."""
    query_embedding = compute_use_embeddings([query_text])[0]
    
    doc_embeddings = compute_use_embeddings([doc_texts[doc_id] for doc_id in initial_results])
    
    scores = {doc_id: float(cosine_similarity([query_embedding], [doc_embeddings[i]])[0][0])
              for i, doc_id in enumerate(initial_results)}
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def load_tfidf_results(results_file):
    """Parses results.txt to extract top 100 document rankings for each query."""
    tfidf_results = {}
    with open(results_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            query_id, _, doc_id, rank = parts[:4]
            rank = int(rank)
            
            if query_id not in tfidf_results:
                tfidf_results[query_id] = []
            
            if rank <= 100:
                tfidf_results[query_id].append(doc_id)
    
    return tfidf_results

def main(tfidf_results_file, corpus_file, queries_file, output_file):
    """Loads data, applies USE re-ranking, and saves results."""
    tfidf_results = load_tfidf_results(tfidf_results_file)
    
    doc_texts = {}
    with open(corpus_file, "r") as file:
        for line in file:
            document = json.loads(line.strip())
            doc_id = document["_id"]
            title = document.get("title", "")
            text = document.get("text", "")
            doc_texts[doc_id] = title + " " + text
    
    with open(queries_file, "r") as file, open(output_file, "w") as out_file:
        for line in file:
            query_data = json.loads(line.strip())
            query_id = query_data["_id"]
            query_text = query_data["text"]
            
            initial_results = tfidf_results.get(query_id, [])
            
            ranked_results = rerank_documents(query_text, initial_results, doc_texts)
            
            for rank, (doc_id, score) in enumerate(ranked_results, start=1):
                out_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} A2-USE-Retrieval\n")

# File paths
tfidf_results_file = "ResultsA1.txt"  # Now using results.txt from Assignment 1
corpus_file = "scifact/corpus.jsonl"
queries_file = "scifact/queries.jsonl"
output_file = "USE_Results.txt"

main(tfidf_results_file, corpus_file, queries_file, output_file)

#trec_eval scifact/qrels/train_fixed.txt USE_Results.txt                                 
#map                     all     0.3524
#gm_map                  all     0.0701
#P_10                    all     0.0602