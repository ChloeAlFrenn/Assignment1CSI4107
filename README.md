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
Prerequisite: Have python installed

*Install Dependencies* - Ensure that you have the required libraries installed:
(If you use python3, use pip3 to install dependencies. For python, use pip)

`pip install sentence-transformers tensorflow tensorflow-hub numpy scikit-learn`

1. BERT-based Retrieval System**: in A2.py

   `python A2.py`
   - Results found in Results.txt (best system) 

2. Universal Sentence Encoder (USE)-based Re-ranking System**: in A2_USE.py

   - run A2_USE.py
   `python A2_USE.py`
   - Results found in USE_Results.txt


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

#### **Evaluation Results Summary**

- **MAP (Mean Average Precision)**: 
   - **BERT-based Retrieval**: 0.6032
   - **USE-based Re-ranking**: 
   - **TF-IDF (Assignment 1)**: 0.5610

- **Precision at 10**:
   - **BERT-based Retrieval**: 0.6032
   - **USE-based Re-ranking**: 
   - **TF-IDF (Assignment 1)**: 0.0857
     
#### **Discussion of Results**

MAP (Mean Average Precision)
- BERT-based Retrieval achieved the highest MAP score of 0.6032, which indicates that the BERT embeddings performed the best when it came to ranking precision across all queries. The high MAP score suggests that BERT's deep learning-based approach was effective at understanding the semantic content of both the queries and documents, resulting in more relevant documents being ranked higher.
BERT’s ability to capture complex contextual relationships between words and phrases enabled it to outperform the TF-IDF method, likely due to its caability of semantic understanding, which allows it to rank documents more accurately even if the query and document do not share many exact terms. 

- The USE-based Re-ranking achieved a lower MAP score of 0.3333, which suggests that although the Universal Sentence Encoder (USE) is capable of capturing semantic similarities, it was less effective than BERT in this case. One possible reason for this could be that USE embeddings, while useful for semantic understanding, may not have captured the nuances of the document-query relationships as well as BERT did. 

- The TF-IDF method from Assignment 1 scored 0.5610, which is significantly lower than BERT-based retrieval but higher than USE-based re-ranking. While TF-IDF is a traditional and efficient retrieval method based on term frequency and inverse document frequency, it doesn’t capture the semantic meaning or context between query and document. As a result, it doesn't rank documents with the same level of precision as neural models like BERT. However, TF-IDF’s performance still ranks higher than the USE-based re-ranking, possibly because the initial retrieval in Assignment 1 was based on a more straightforward keyword match.
