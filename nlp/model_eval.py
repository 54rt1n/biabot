#%%
import numpy as np
from typing import List, Tuple, Dict
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd

# List of tokenizers to evaluate
BPE_TOKENIZERS = [
    "openai-gpt",
    "t5-base",
    "roberta-base",
    "facebook/bart-base",
    "xlnet-base-cased",
    "bert-base-uncased",
    "albert-base-v2",
    "distilbert-base-uncased",
    "google/electra-small-generator",
    "ctrl"
]

# Sample corpus (unchanged)
CORPUS = [
    "Artificial General Intelligence (AGI) is a hypothetical type of intelligent agent.",
    "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and AI.",
    "Machine Learning (ML) algorithms build a model based on sample data.",
    "The Turing test, developed by Alan Turing in 1950, is a test of a machine's ability to exhibit intelligent behavior.",
    "Deep Learning is part of a broader family of ML methods based on artificial neural networks.",
    "Reinforcement Learning (RL) is an area of ML concerned with how software agents ought to take actions in an environment.",
    "Computer Vision is an interdisciplinary scientific field that deals with how computers gain high-level understanding from digital images or videos.",
    "The Internet of Things (IoT) describes physical objects with sensors, processing ability, software, and other technologies.",
    "Quantum Computing is the use of quantum phenomena such as superposition and entanglement to perform computation.",
    "Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks."
]

# Revised queries with explicit evaluation targets
QUERIES = [
    {
        "query": "What is the relationship between AGI and ML?",
        "should_match": [0, 2],  # Indices of CORPUS that should match
        "shouldnt_match": [7, 8, 9]  # Indices of CORPUS that shouldn't match
    },
    {
        "query": "How does NLP relate to Deep Learning in AI?",
        "should_match": [1, 4],
        "shouldnt_match": [3, 7, 9]
    },
    {
        "query": "Explain the connection between RL and IoT in Cybersecurity.",
        "should_match": [5, 7, 9],
        "shouldnt_match": [0, 3, 8]
    },
    {
        "query": "What are the applications of Computer Vision in IoT?",
        "should_match": [6, 7],
        "shouldnt_match": [0, 2, 9]
    },
    {
        "query": "How does Quantum Computing differ from classical Machine Learning?",
        "should_match": [2, 8],
        "shouldnt_match": [1, 6, 9]
    }
]

def tokenize_text(tokenizer, text: str) -> List[str]:
    return tokenizer.tokenize(text)

def compute_f1_score(relevant_tokens: set, retrieved_tokens: set) -> float:
    true_positives = len(relevant_tokens.intersection(retrieved_tokens))
    if len(retrieved_tokens) == 0:
        precision = 0
    else:
        precision = true_positives / len(retrieved_tokens)
    
    if len(relevant_tokens) == 0:
        recall = 0
    else:
        recall = true_positives / len(relevant_tokens)
    
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_tokenizer(tokenizer_name: str) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    corpus_tokens = [set(tokenize_text(tokenizer, doc)) for doc in CORPUS]
    
    results = {}
    for i, query_data in enumerate(QUERIES):
        query_tokens = set(tokenize_text(tokenizer, query_data["query"]))
        
        # Compute F1 scores for documents that should match
        should_f1s = [compute_f1_score(query_tokens, corpus_tokens[j]) for j in query_data["should_match"]]
        avg_should_f1 = np.mean(should_f1s)
        
        # Compute F1 scores for documents that shouldn't match
        shouldnt_f1s = [compute_f1_score(query_tokens, corpus_tokens[j]) for j in query_data["shouldnt_match"]]
        avg_shouldnt_f1 = np.mean(shouldnt_f1s)
        
        # Compute a combined score (higher is better)
        combined_score = avg_should_f1 - avg_shouldnt_f1
        
        results[f"Query_{i+1}_Should"] = avg_should_f1
        results[f"Query_{i+1}_Shouldnt"] = avg_shouldnt_f1
        results[f"Query_{i+1}_Combined"] = combined_score
    
    results["Average_Combined"] = np.mean([results[k] for k in results if k.endswith("Combined")])
    
    return results

# Evaluate tokenizers
results = {}
for tokenizer_name in tqdm(BPE_TOKENIZERS, desc="Evaluating tokenizers"):
    results[tokenizer_name] = evaluate_tokenizer(tokenizer_name)

# Create and display results DataFrame
df_results = pd.DataFrame(results).T
df_results = df_results.sort_values("Average_Combined", ascending=False)
df_results


# %%
