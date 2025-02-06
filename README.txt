Information Retrieval Assignment 1

Team member contribution:
- Chloé Al-Frenn (300211508): Implemented Step 1 
- Anoushka Jawale (300233148) : Implemented Step 2
- Yasmine Zoubdi (300170464) : Implemented Step 3
  
Functionnality overview of the programm:
    This project implements an Information Retrieval (IR) system using the vector space model for retrieving relevant scientific abstracts based on given queries. The system processes the Scifact dataset, builds an inverted index, and ranks documents using TF-IDF weighted cosine similarity.
    The system consists of three main steps:
        1. Preprocessing: Tokenization, stopword removal, optional stemming
        2. Indexing: Building an inverted index
        3. Retrieval and Ranking: Query processing, document retrieval using cosine similarity, and ranking

Instructions for running the Program:
    Prerequisites:

        - Ensure you have Python installed
        - Ensure you have these libraries intalled: 
            - json
            - re
            - math
            - collections
            - nltk (to install the nltk library you can use "pip install nltk")

    To run the program:

        1. Verify that the following files are in the working directory:
            - `scifact/corpus.jsonl` (Document corpus)
            - `scifact/queries.jsonl` (Query set)
            - `stopwords.txt` (List of stopwords)
     
        2. Run the program with: "python A1.py"

        3. The output of the program will be avaialble in the Results.txt file

Algorithms, Data Structures & Optimizations

    Step 1: Preprocessing
        This step begins by loading the stopwords from the stopwords.txt file into a set. The preprocessing follows these steps:
            First it removes unwanted characters using a regular expression, keeping only letters and spaces.
            Then it converts all remaining characters to lowercase and splits the text into individual words to get them as tokens.
            It will then compare the tokens to the stopword set and removes matching words.
            Finally it can Optionally apply stemming using the porter Stemmer from the nltk library to reduce words to their root form.
       
        Optimization:
            Using a set for stopwords enables O(1) lookup time, making the filtering process significantly faster.
            The regex is optimized to remove non-letter characters in a single pass, reducing processing time.
            Stemming is left as an option, allowing flexibility for performance vs. accuracy trade-offs.

    Step 2: Indexing
        This step constructs an inverted index from the preprocessed text, allowing for efficient document retrieval. The process follows these steps:
            First it builds the Inverted Index by reading each document from the dataset, extracting the document ID and processed tokens.
            For each token, it will update the index by recording the document ID and term frequency (TF) (the number of times the term appears in the document).
            Then it will use a defaultdict of dictionaries (defaultdict(lambda: defaultdict(int))) to store the index efficiently.
            The structure is {term: {doc_id: term_frequency}}, ensuring fast access to term occurrences in documents.
            
        Optimization:
            Using defaultdict reduces the need for explicit key checks, improving speed.
            Each document is processed only once, minimizing redundant operations.
            Instead of storing full documents, only term occurrences are stored, reducing memory usage.
    
    Step 3: Retrieval and Ranking
        This step retrieves relevant documents for a given query and ranks them based on cosine similarity using TF-IDF weighting. The process follows these steps:
            First the query is tokenized and preprocessed using the same methods as document preprocessing.
            then a query vector is built, where term weights are calculated using TF-IDF.
            For each query term, it will retrieve the relevant documents from the inverted index and compute TF-IDF weights for query terms and document terms.
            It will then calculate the dot product between the query and document vectors and normalizes scores using the L2 norm of both the query and document vectors.
            For ranking and the output the documents are ranked in descending order of similarity scores.
            The top 100 results for each query are saved in the required format.

        Optimization:
            The L2 norm of document vectors is precomputed, reducing redundant calculations at query time.
            The system only considers documents that contain at least one query term, avoiding unnecessary computations.
            Uses Python’s optimized sorted() function to rank documents efficiently.
Obtained Results:
    
    Vocabulary:
       With and Without using the PorterStemmer the size of the vocabulary was of 590739 tokens. 
       Some words seem to repeat themselves like "human" adding a method to the preprocessing that removes duplicates might make the program more efficient.

    Sample vocabulary without the stemmer: 
        alterations
        architecture
        cerebral
        white
        matter
        developing
        human
        brain
        affect
        cortical
        development
        result
        functional
        disabilities
        line
        scan
        diffusionweighted
        magnetic
        resonance
        imaging
        mri
        sequence
        diffusion
        tensor
        analysis
        applied
        measure
        apparent
        diffusion
        coefficient
        calculate
        relative
        anisotropy
        delineate
        threedimensional
        fiber
        architecture
        cerebral
        white
        matter
        preterm
        fullterm
        infants
        assess
        effects
        prematurity
        cerebral
        white
        matter
        development
        early
        gestation
        preterm
        infants
        studied
        term
        central
        white
        matter
        mean
        apparent
        diffusion
        coefficient
        wk
        micromms
        decreased
        term
        micromms
        posterior
        limb
        internal
        capsule
        mean
        apparent
        diffusion
        coefficients
        versus
        micromms
        relative
        anisotropy
        closer
        birth
        term
        absolute
        values
        internal
        capsule
        central
        white
        matter
        preterm
        infants
        term
        mean
        diffusion
        coefficients
        central
        white
        matter
        versus

    Sample vocabulary with the stemmer: 
        alter
        architectur
        cerebr
        white
        matter
        develop
        human
        brain
        affect
        cortic
        develop
        result
        function
        disabl
        line
        scan
        diffusionweight
        magnet
        reson
        imag
        mri
        sequenc
        diffus
        tensor
        analysi
        appli
        measur
        appar
        diffus
        coeffici
        calcul
        rel
        anisotropi
        delin
        threedimension
        fiber
        architectur
        cerebr
        white
        matter
        preterm
        fullterm
        infant
        assess
        effect
        prematur
        cerebr
        white
        matter
        develop
        earli
        gestat
        preterm
        infant
        studi
        term
        central
        white
        matter
        mean
        appar
        diffus
        coeffici
        wk
        micromm
        decreas
        term
        micromm
        posterior
        limb
        intern
        capsul
        mean
        appar
        diffus
        coeffici
        versu
        micromm
        rel
        anisotropi
        closer
        birth
        term
        absolut
        valu
        intern
        capsul
        central
        white
        matter
        preterm
        infant
        term
        mean
        diffus
        coeffici
        central
        white
        matter
        versu

    Queries result: 
        Using only the title seems to give a better score. Overall, the system was able to identify relevant documents, but further refinements, such as expanding queries or adjusting weighting schemes, could enhance retrieval performance.

    Sample queries output using only the title:   
        0 Q0 25602549 1 0.2886 Assignment1
        0 Q0 42421723 2 0.2868 Assignment1
        0 Q0 16532419 3 0.2579 Assignment1
        0 Q0 14831629 4 0.2502 Assignment1
        0 Q0 18953920 5 0.2465 Assignment1
        0 Q0 11349166 6 0.2451 Assignment1
        0 Q0 22972632 7 0.2451 Assignment1
        0 Q0 4405194 8 0.2434 Assignment1
        0 Q0 13938878 9 0.2390 Assignment1
        0 Q0 9550981 10 0.2266 Assignment1

        2 Q0 4828631 1 0.3003 Assignment1
        2 Q0 103007 2 0.2700 Assignment1
        2 Q0 7020505 3 0.2693 Assignment1
        2 Q0 1583134 4 0.2486 Assignment1
        2 Q0 16398049 5 0.2197 Assignment1
        2 Q0 34733465 6 0.2167 Assignment1
        2 Q0 11880289 7 0.2161 Assignment1
        2 Q0 32481310 8 0.2158 Assignment1
        2 Q0 24347647 9 0.1920 Assignment1
        2 Q0 13883546 10 0.1915 Assignment1

    Sample queries output using the title and text:
        0 Q0 13231899 1 0.0803 Assignment1
        0 Q0 1836154 2 0.0764 Assignment1
        0 Q0 42373087 3 0.0724 Assignment1
        0 Q0 42421723 4 0.0648 Assignment1
        0 Q0 10906636 5 0.0626 Assignment1
        0 Q0 994800 6 0.0551 Assignment1
        0 Q0 35008773 7 0.0498 Assignment1
        0 Q0 12156187 8 0.0498 Assignment1
        0 Q0 14827874 9 0.0468 Assignment1
        0 Q0 7581911 10 0.0459 Assignment1

        2 Q0 13734012 1 0.2775 Assignment1
        2 Q0 26059876 2 0.1865 Assignment1
        2 Q0 11880289 3 0.1320 Assignment1
        2 Q0 14610165 4 0.1211 Assignment1
        2 Q0 103007 5 0.1205 Assignment1
        2 Q0 18340282 6 0.0875 Assignment1
        2 Q0 16398049 7 0.0798 Assignment1
        2 Q0 15327601 8 0.0777 Assignment1
        2 Q0 32922179 9 0.0698 Assignment1
        2 Q0 32481310 10 0.0679 Assignment1


Mean Average Precision (MAP) score:
    Using trec_eval the mean average precision score we got was 0.5216 this was calculated using both the title and the text of the queries.

