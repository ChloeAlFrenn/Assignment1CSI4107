### **Information Retreival and the Internet Assignment 2**

#### **Team Information**
   - Yasmine Zoubdi (300170464): Responsible for implementing and testing the **BERT-based retrieval system**.
   - Chloé Al-Frenn (300211508): Responsible for implementing and testing the **Universal Sentence Encoder (USE)-based re-ranking system**.
   - Anoushka Jawale (300233148): Responsible for report of results 

#### **Functionality of the Programs**

The goal of this assignment was to improve the Information Retrieval (IR) system using advanced neural models such as **BERT** and **Universal Sentence Encoder (USE)**, and then evaluate them to achieve better results than the **TF-IDF-based system** used in Assignment 1.

##### **Program Overview**
The program consists of two key components:

1. **BERT-based Retrieval System**: in A2.py
   - Takes a corpus of documents and test queries in JSON format as input 
   - Uses a pre-trained **BERT** model (`all-MiniLM-L6-v2`) to generate embeddings for documents and queries, which are calculated and kept in a dictionary.
   - For each query, the cosine similarity between the query's embedding and each document's embedding is calculated.
   - The documents are then ranked based on similarity scores.
   - Outputs a list of the documents, ranked. 

3. **Universal Sentence Encoder (USE)-based Re-ranking System**: in A2_USE.py
   - Takes a corpus of documents and test queries in JSON format as input 
   - Re-ranks the documents (retrieved via the original TF-IDF approach from Assignment 1) using **USE embeddings**.
   - Embeddings for both the query and the documents are generated using the **Universal Sentence Encoder**.
   - The cosine similarity is used to calculate the re-ranking scores, and the results are outputted as the re-ranked document list.
   - Outouts a re-ranked list of the documents.

##### **How to Run the Programs**

Install Dependencies: Ensure that you have the required libraries installed:

`pip install sentence-transformers tensorflow tensorflow-hub numpy scikit-learn`

1. BERT-based Retrieval System**: in A2.py

   - `python A2.py`
   - Results found in Results.txt (best system) 

2. Universal Sentence Encoder (USE)-based Re-ranking System**: in A2_USE.py

   - first run A1.PY
   - `python A1.py`
   - then run A2_USE.py
   - `python A2_USE.py`
   - Results found in USE_Results.txt
  
3. Evaluating the Models
   - `trec_eval scifact/qrels/train_fixed.txt Results.txt` for BERT-based Retrieval System
   - `trec_eval scifact/qrels/train_fixed.txt USE_Results.txt` for Universal Sentence Encoder (USE)-based Re-ranking System
  


#### **Algorithms and Data Structures Used**
- **BERT-based System**:
   - **SentenceTransformer** is used to load the pre-trained BERT model (`all-MiniLM-L6-v2`).
   - **Cosine Similarity** is used to measure the similarity between query and document embeddings.
   - **Dictionary**: A dictionary (`doc_embeddings`) is used to store document embeddings, where the key is the document ID and the value is the corresponding embedding.

- **USE-based Re-ranking System**:
   - **Universal Sentence Encoder (USE)** is used for generating embeddings for both queries and documents.
   - **Cosine Similarity** is again used for re-ranking the documents.
   - **Dictionary**: The document texts are stored in a dictionary (`doc_texts`) where the key is the document ID and the value is the combined text (title + content).
   - **TF-IDF Results**: The results of the initial retrieval from Assignment 1 are parsed and used as the input for re-ranking.

#### **First 10 Answers to Queries **


#### **Discussion of Results**
The system was evaluated using two different neural retrieval methods: **BERT** and **USE**. The results were compared to the initial TF-IDF-based retrieval results from Assignment 1.

- **BERT-based Retrieval**: The BERT embeddings provided high-quality semantic understanding of the documents and queries, resulting in more relevant retrieval compared to the TF-IDF-based approach. The ranking was sensitive to semantic nuances in the query and document content.

- **USE-based Re-ranking**: The USE embeddings provided a semantic re-ranking of the results obtained from the TF-IDF approach. The re-ranking did not improve the precision and relevance of the top-ranked documents, as measured by evaluation metrics such as MAP and P@10.

#### **Evaluation Results**

- **MAP (Mean Average Precision)**: 
   - **BERT-based Retrieval**: 0.6032
   - **USE-based Re-ranking**: 
   - **TF-IDF (Assignment 1)**: 0.50

#### **Discussion of Results**



