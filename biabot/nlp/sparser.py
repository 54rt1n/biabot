import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from typing import List

BPE_TOKENIZERS = [
    "openai-gpt",
    "t5-base",
    "roberta-base",
    "facebook/bart-base",
    "xlnet-base-cased",
    "bert-base-uncased",
    "albert-base-v2",
    "distilbert-base-uncased",
    "ctrl",
    "google/electra-small-generator",
]


class SparseVectorizer:
    """
    This is a BPE-based implementation of the BM25 algorithm for sparse vectorization.

    BM25 (Best Matching 25):
    BM25 is an advanced ranking function used in information retrieval, particularly in search engines.
    It improves upon the basic TF-IDF (Term Frequency-Inverse Document Frequency) model by incorporating
    several key features:

    1. Document and Query Term Weighting: BM25 considers both the frequency of terms in a document
       and how rare those terms are across all documents in the corpus.
    2. Length Normalization: BM25 takes into account the length of documents, preventing bias towards
       longer documents that might naturally contain more occurrences of search terms.
    3. Saturation Function: BM25 uses a saturation function to limit the impact of term frequency,
       meaning that after a certain point, additional occurrences of a term in a document don't
       significantly increase the document's relevance score.

    The BM25 scoring function is defined as:
    score(D, Q) = sum(IDF(q_i) * ((k1 + 1) * f(q_i, D)) / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl)))

    Where:
    - D is a document
    - Q is a query
    - q_i is a query term
    - f(q_i, D) is the frequency of q_i in D
    - |D| is the length of document D
    - avgdl is the average document length in the corpus
    - k1 and b are free parameters

    BPE (Byte Pair Encoding):
    BPE is a data compression technique that has found significant use in natural language processing,
    particularly in tokenization for large language models. The process works as follows:

    1. Start with a vocabulary of individual characters.
    2. Count the frequency of adjacent pairs of symbols (initially characters) in the training data.
    3. Replace the most frequent pair with a new symbol.
    4. Repeat steps 2-3 for a fixed number of iterations or until a desired vocabulary size is reached.

    BPE allows for a balance between character-level and word-level tokenization:
    - It can handle out-of-vocabulary words by breaking them into subword units.
    - Common words remain as single tokens, while rarer words are split into more common subword units.
    - It's effective for morphologically rich languages and can handle different variations of words.

    BPE and Lemmatization:
    An important advantage of BPE tokenization is that it often reduces or eliminates the need for 
    explicit lemmatization. Lemmatization is the process of reducing words to their base or dictionary 
    form (e.g., "running" to "run", "better" to "good"). BPE achieves a similar effect through its 
    subword tokenization:

    1. Morphological Variants: BPE can break words into stems and affixes. For example, "running" 
       might be tokenized as "run" + "##ing", effectively capturing the root "run".
    2. Irregular Forms: While not perfect, BPE can often capture relationships between irregular 
       forms (like "better" and "good") by breaking them into shared subwords.
    3. Out-of-vocabulary Words: BPE can handle unseen words by breaking them into known subwords, 
       often preserving the semantic relationship to their lemmas.
    4. Language Agnostic: Unlike many lemmatization algorithms, BPE doesn't rely on language-specific 
       rules, making it more versatile across different languages.

    This subword tokenization allows the model to implicitly learn and utilize morphological 
    relationships, often making explicit lemmatization unnecessary. However, for some specific NLP 
    tasks or languages, combining BPE with lemmatization might still provide additional benefits.

    This implementation uses a BPE tokenizer (via the Hugging Face Transformers library) to tokenize
    the input text, and then applies the BM25 algorithm for scoring and retrieval.
    """
    def __init__(self, model_name: str = 'bert-base-uncased', k1: float = 1.5, b: float = 0.75):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.k1 = k1
        self.b = b
        self.vectorizer = self._create_vectorizer()
        self.idf = None
        self.doc_len = None
        self.avgdl = None

    def _create_vectorizer(self):
        """
        Creates a TfidfVectorizer instance with the specified configuration.
        
        The TfidfVectorizer is used to transform text documents into sparse feature vectors, where each feature corresponds to a unique token in the corpus. The IDF (Inverse Document Frequency) weighting scheme is used to downscale the importance of common tokens.
        
        This method configures the TfidfVectorizer with the following settings:
        - `tokenizer`: Uses the `tokenize` method of the current `SparseVectorizer` instance to tokenize the input text.
        - `lowercase`: Converts all characters to lowercase before tokenization.
        - `use_idf`: Enables the use of Inverse Document Frequency (IDF) weighting.
        - `norm`: Disables normalization of the feature vectors.
        - `smooth_idf`: Disables smoothing of the IDF weights.
        """
                
        return TfidfVectorizer(
            tokenizer=self.tokenize,
            lowercase=True,
            use_idf=True,
            norm=None,
            smooth_idf=False
        )

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the given text using the configured tokenizer.
        
        Args:
            text (str): The input text to be tokenized.
        
        Returns:
            List[str]: A list of tokens representing the tokenized text.
        """

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [self.tokenizer.decode([id]) for id in token_ids]

    def clear(self):
        """
        Resets the internal state of the SparseVectorizer instance, including the IDF weights, document lengths, and average document length. This also recreates the TfidfVectorizer instance used for text vectorization.
        """
        self.idf = None
        self.doc_len = None
        self.avgdl = None
        self.vectorizer = self._create_vectorizer()

    @property
    def feature_names(self) -> List[str]:
        """
        Returns a list of feature names (unique tokens) that the SparseVectorizer has learned from the input documents.
        
        Returns:
            List[str]: A list of feature names (unique tokens) used by the SparseVectorizer.
        """
                
        return self.vectorizer.get_feature_names_out()

    def fit(self, documents: List[str]):
        """
        Fits the SparseVectorizer to the provided documents and calculates the BM25 weights for the document corpus.
        
        This method performs the following steps:
        1. Fits the TfidfVectorizer to the input documents and retrieves the IDF (Inverse Document Frequency) weights.
        2. Converts the sparse document-term matrix to a dense PyTorch tensor and calculates the length of each document.
        3. Computes the average document length across the corpus.
        4. Calculates the BM25 (Okapi BM25) weights for the document corpus using the computed document lengths and the IDF weights.
        5. Stores the calculated BM25 weights in the `bm25_weights` attribute.
        
        Args:
            documents (List[str]): A list of document strings to be used for fitting the SparseVectorizer.
        
        Returns:
            SparseVectorizer: The fitted SparseVectorizer instance.
        """
        
        X = self.vectorizer.fit_transform(documents)
        self.idf = torch.FloatTensor(self.vectorizer.idf_)
        X_torch = torch.FloatTensor(X.toarray())
        self.doc_len = X_torch.sum(dim=1)
        self.avgdl = self.doc_len.mean()
        
        # Calculate BM25 weights for the document corpus
        
        len_d = self.doc_len.unsqueeze(1)
        numerator = X_torch * (self.idf * (self.k1 + 1)).unsqueeze(0)
        denominator = X_torch * (self.k1 * (1 - self.b + self.b * len_d / self.avgdl)) + 1.0
        self.bm25_weights = numerator / denominator
        
        return self

    def transform(self, documents: List[str]) -> torch.Tensor:
        """
        Transforms the input documents into a tensor of BM25-weighted document vectors.
        
        This method performs the following steps:
        1. Transforms the input documents into a sparse document-term matrix using the fitted `vectorizer`.
        2. Converts the sparse matrix to a dense PyTorch tensor.
        3. Applies the pre-computed IDF (Inverse Document Frequency) weights to the document vectors.
        4. Applies the BM25 (Okapi BM25) weighting factors to the document vectors.
        
        The resulting tensor contains the BM25-weighted document vectors, which can be used for similarity computations or other downstream tasks.
        
        Args:
            documents (List[str]): A list of document strings to be transformed.
        
        Returns:
            torch.Tensor: A tensor of BM25-weighted document vectors.
        """
                
        X = self.vectorizer.transform(documents)
        X_torch = torch.FloatTensor(X.toarray())
        
        # Apply IDF weights only, as BM25 document-specific factors are pre-computed
        vectors = X_torch * self.idf.unsqueeze(0)

        # Apply BM25 factors
        vectors = vectors * (self.k1 + 1) / (vectors + self.k1)

        return vectors

    def query(self, queries: List[str]) -> torch.Tensor:
        """
        Computes the similarity between the input queries and the pre-weighted document vectors.
        
        This method performs the following steps:
        1. Transforms the input queries into BM25-weighted query vectors using the `transform` method.
        2. Computes the dot product between the query vectors and the pre-computed BM25-weighted document vectors.
        3. Normalizes the query and document vectors by their respective L2 norms.
        4. Computes the cosine similarity between the query and document vectors.
        5. Replaces any NaN values in the similarity matrix with zeros.
        
        The resulting similarity matrix contains the cosine similarity scores between each query and each document, which can be used for ranking or other downstream tasks.
        
        Args:
            queries (List[str]): A list of query strings to be compared against the documents.
        
        Returns:
            torch.Tensor: A tensor of cosine similarity scores between the queries and documents.
        """
                
        query_vectors = self.transform(queries)
        
        # Compute similarity with pre-weighted documents
        dot_product = torch.mm(query_vectors, self.bm25_weights.t())
        query_norm = torch.norm(query_vectors, dim=1).unsqueeze(1)
        doc_norm = torch.norm(self.bm25_weights, dim=1).unsqueeze(0)
        
        similarity = dot_product / (query_norm * doc_norm)
        
        # Replace NaN values with zeros
        similarity = torch.nan_to_num(similarity, nan=0.0)
        
        return similarity

    @classmethod
    def from_documents(cls, documents: List[str], model_name: str = 'ctrl'):
        """
        Transforms a list of documents into a sparse matrix of BM25-weighted document vectors.
        
        Args:
            documents (List[str]): A list of document strings to be transformed.
            model_name (str, optional): The name of the language model to use for the transformation. Defaults to 'ctrl'.
        
        Returns:
            torch.Tensor: A tensor of BM25-weighted document vectors.
        """
                
        vectorizer = cls(model_name=model_name)
        vectorizer.fit(documents)
        return vectorizer.transform(documents)

    @classmethod
    def fit_transform_query(cls, documents: List[str], queries: List[str], model_name: str = 'bert-base-uncased'):
        """
        Fits a SparseVectorizer to the provided documents, and then uses it to compute the similarity between the provided queries and the documents.
        
        Args:
            documents (List[str]): A list of document strings to be transformed.
            queries (List[str]): A list of query strings to be compared against the documents.
            model_name (str, optional): The name of the language model to use for the transformation. Defaults to 'ctrl'.
        
        Returns:
            torch.Tensor: A tensor of cosine similarity scores between the queries and documents.
        """

        vectorizer = cls(model_name=model_name)
        vectorizer.fit(documents)
        return vectorizer.query(queries, documents)

def test():
    documents = [
        "Artificial General Intelligence is a subfield of Machine Learning.",
        "Natural Language Processing is crucial for AGI development.",
        "The AGI researcher published a paper on advanced NLP techniques."
    ]
    queries = ["What is the relationship between AGI and NLP?", "Why do I care about intelligence?"]

    results = {}
    features = {}
    shapes = {}
    # Example usage:
    encoder = 'bert-base-uncased'
    vectorizer = SparseVectorizer(model_name=encoder)

    vectorizer.fit(documents)
    features[encoder] = vectorizer.feature_names
    vector_matrix = vectorizer.transform(documents)
    shapes[encoder] = vector_matrix.shape

    #%%
    # Scoring example
    vectorizer.query(queries)

    # %%
