import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained BERT model for embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_bert_embeddings(corpus_file):
    """
    Creates document embeddings using BERT instead of TF-IDF.
    """
    doc_embeddings = {}
    doc_texts = {}

    with open(corpus_file, "r") as file:
        for line in file:
            document = json.loads(line.strip())
            doc_id = document["_id"]
            title = document.get("title", "")
            text = document.get("text", "")
            full_text = title + " " + text

            doc_texts[doc_id] = full_text 

    # Compute embeddings in batch 
    doc_ids = list(doc_texts.keys())
    doc_sentences = list(doc_texts.values())
    embeddings = bert_model.encode(doc_sentences, convert_to_numpy=True)

    # Store embeddings in a dictionary
    for i, doc_id in enumerate(doc_ids):
        doc_embeddings[doc_id] = embeddings[i]

    return doc_embeddings



def compute_bert_similarity(query_text, doc_embeddings):
    """
    Computes cosine similarity between query and all documents using BERT embeddings.
    """
    query_embedding = bert_model.encode([query_text], convert_to_numpy=True)[0]  # Query vector
    scores = {}

    for doc_id, doc_embedding in doc_embeddings.items():
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]  # Compute cosine similarity
        scores[doc_id] = similarity

    # Return sorted results
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def main(corpus_file, queries_file, output_file):
    # Generate document embeddings using BERT
    doc_embeddings = create_bert_embeddings(corpus_file)

    # Process queries and write results
    with open(queries_file, "r") as file, open(output_file, "w") as out_file:
        for line in file:
            query_data = json.loads(line.strip())
            query_id = query_data["_id"]
            query_text = query_data["text"]

            # Compute similarity with BERT
            ranked_results = compute_bert_similarity(query_text, doc_embeddings)

            # Write top 100 results
            for rank, (doc_id, score) in enumerate(ranked_results[:100], start=1):
                out_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} A2-BERT-Retrieval\n")

corpus_file = "scifact/corpus.jsonl" 
queries_file = "scifact/queries.jsonl" 
output_file = "Results.txt"     

main(corpus_file, queries_file, output_file)