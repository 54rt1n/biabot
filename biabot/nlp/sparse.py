# biabot/nlp/sparse.py

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Optional

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


import re

def find_capitalized_ngrams(text: str, max_n: int = 3) -> List[str]:
    """
    Find capitalized 1-grams, 2-grams, and 3-grams in the text.

    Args:
        text (str): The input text.
        max_n (int): The maximum number of consecutive capitalized words to consider as an n-gram.

    Returns:
        List[str]: A list of matched capitalized n-grams.
    """
    # Patterns to match
    mixed_case_pattern = re.compile(r'\b[A-Z][a-z]*[A-Z][a-z]*\b')
    capitalized_word_pattern = re.compile(r'\b[A-Z][a-z]+\b')

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    matches = []

    for sentence in sentences:
        # Tokenize words, remove punctuation
        words = re.findall(r'\b\w+\b', sentence)
        if not words:
            continue

        idx = 0
        while idx < len(words):
            word = words[idx]

            # Check if the word is mixed-case
            if mixed_case_pattern.match(word):
                matches.append(word)

            # Handle sentence-initial word
            if idx == 0:
                idx += 1
                continue

            # Check for capitalized n-grams (excluding sentence-initial word)
            ngram_words = []
            n = 0
            while n < max_n and idx + n < len(words):
                next_word = words[idx + n]
                if capitalized_word_pattern.match(next_word):
                    ngram_words.append(next_word)
                    n += 1
                else:
                    break

            if ngram_words:
                matches.append(' '.join(ngram_words))
                idx += n
            else:
                idx += 1

    # Remove duplicates
    matches = set(matches)
    #print(f"Matches: {matches}")
    
    # Add singular and/or plural
    for match in list(matches):
        singular_match = match.rstrip('s')
        # TODO add es's
        plural_match = match + 's'
        if singular_match in matches:
            matches.add(plural_match)
            #print(f"Added plural: {plural_match}")
        if plural_match in matches:
            matches.add(singular_match)
            #print(f"Added singular: {singular_match}")

    return list(matches)

class SparseVectorizer:
    """
    The SparseVectorizer class is designed to convert text data into sparse vector representations.
    
    It uses the Bag of Words (BoW) approach, where each document is represented as a vector of word frequencies,
    and it uses the TF-IDF (Term Frequency-Inverse Document Frequency) weighting scheme.
    
    We tokenize the text using the NLTK library, and apply lemmatization and stop word removal.
    """
    
    def __init__(self, ner_boost: int = 4):
        """
        Initializes the SparseVectorizer class with the specified NER boost value.
        
        Args:
            ner_boost (int): The boost factor to apply to named entity recognition (NER) terms.
        
        Attributes:
            lemmatizer (WordNetLemmatizer): A WordNetLemmatizer instance used for lemmatization.
            stop_words (set): A set of English stop words.
            vectorizer (TfidfVectorizer): A TfidfVectorizer instance used for text vectorization.
            lemmatize (bool): A flag indicating whether to perform lemmatization.
            ner_boost (int): The boost factor to apply to named entity recognition (NER) terms.
            ner_terms (set): A set of named entity recognition (NER) terms.
        """
                
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocess_text,
            token_pattern=None,
            lowercase=False,
            min_df=1,
            ngram_range=(1, 3),
        )
        self.lemmatize = False
        self.ner_boost = ner_boost
        self.ner_terms = set()

    def get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """
        Converts a part-of-speech tag from the Penn Treebank tagset to the corresponding WordNet part-of-speech tag.

        Args:
            treebank_tag (str): The part-of-speech tag from the Penn Treebank tagset.

        Returns:
            The corresponding WordNet part-of-speech tag.
        """
        tag_dict = {
            'CC': wordnet.ADV,   # Coordinating conjunction
            'CD': None,          # Cardinal number
            'DT': wordnet.ADV,   # Determiner
            'EX': wordnet.ADV,   # Existential there
            'FW': None,          # Foreign word
            'IN': wordnet.ADV,   # Preposition or subordinating conjunction
            'JJ': wordnet.ADJ,   # Adjective
            'JJR': wordnet.ADJ,  # Adjective, comparative
            'JJS': wordnet.ADJ,  # Adjective, superlative
            'LS': None,          # List item marker
            'MD': wordnet.VERB,  # Modal
            'NN': wordnet.NOUN,  # Noun, singular or mass
            'NNS': wordnet.NOUN, # Noun, plural
            'NNP': wordnet.NOUN, # Proper noun, singular
            'NNPS': wordnet.NOUN,# Proper noun, plural
            'PDT': wordnet.ADV,  # Predeterminer
            'POS': wordnet.ADV,  # Possessive ending
            'PRP': wordnet.NOUN, # Personal pronoun
            'PRP$': wordnet.NOUN,# Possessive pronoun
            'RB': wordnet.ADV,   # Adverb
            'RBR': wordnet.ADV,  # Adverb, comparative
            'RBS': wordnet.ADV,  # Adverb, superlative
            'RP': wordnet.ADV,   # Particle
            'SYM': None,         # Symbol
            'TO': wordnet.ADV,   # to
            'UH': None,          # Interjection
            'VB': wordnet.VERB,  # Verb, base form
            'VBD': wordnet.VERB, # Verb, past tense
            'VBG': wordnet.VERB, # Verb, gerund or present participle
            'VBN': wordnet.VERB, # Verb, past participle
            'VBP': wordnet.VERB, # Verb, non-3rd person singular present
            'VBZ': wordnet.VERB, # Verb, 3rd person singular present
            'WDT': wordnet.ADV,  # Wh-determiner
            'WP': wordnet.NOUN,  # Wh-pronoun
            'WP$': wordnet.NOUN, # Possessive wh-pronoun
            'WRB': wordnet.ADV,  # Wh-adverb
        }

        if treebank_tag not in tag_dict:
            # If it is all alpha chars, lets log it, otherwise it's probably junk
            if treebank_tag.isalpha():
                print(f"Unknown POS tag: {treebank_tag}, using NOUN")
            return wordnet.NOUN
        return tag_dict.get(treebank_tag, wordnet.NOUN)


    def preprocess_text(self, text: str):
        """
        Preprocesses the given text by tokenizing it, lemmatizing the tokens, and removing stop words and punctuation.
        
        Args:
            text (str): The text to preprocess.
        
        Returns:
            List[str]: The preprocessed tokens.
        """

        # Clean our text, removing all non-alphanumeric/dash/punnctuation characters
        text = re.sub(r'[^a-zA-Z0-9\s\-\.\,\?\!\:\;]', '', text)
                
        tokens = nltk.word_tokenize(text)
        tagged_tokens = nltk.pos_tag(tokens)

        lemmas = []
        for token, pos in tagged_tokens:
            if token not in self.stop_words and token not in string.punctuation:
                if self.lemmatize:
                    lpos = self.get_wordnet_pos(pos)
                    if lpos is not None:
                        lemmas.append(self.lemmatizer.lemmatize(token, lpos))
                else:
                    lemmas.append(token)

            # We want to boost the NER terms
            if token in self.ner_terms:
                lemmas.extend([token] * self.ner_boost)
        
        return lemmas

    def score_documents(self, queries: List[str], documents: List[str]) -> np.ndarray:
        """
        Scores a list of documents against a given query.

        Args:
            queries (List[str]): The list of queries to score.
            documents (List[str]): The list of document contents to score.
            decay (float): The decay factor to use for temporal decay.

        Returns:
            np.ndarray: An array of scores, where each element corresponds to the score of the corresponding document in the `contents` list.
        """
        self.ner_terms = set([term for query in queries for term in find_capitalized_ngrams(query)])
        self.vectorizer.fit(documents)
        query_vectors = self.vectorizer.transform(queries)
        document_vectors = self.vectorizer.transform(documents)

        # Compute cosine similarity using sklearn's function
        similarities = cosine_similarity(query_vectors, document_vectors)
        return similarities
            