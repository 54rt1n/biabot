# Advancing AGI Through Self-RAG and Active Memory Injection

## Abstract

This white paper outlines the current state of an innovative AGI research application designed to implement Self-RAG (Retrieval-Augmented Generation), LLM self-generation, and active injection of relevant memories into the context. The project aims to push the boundaries of artificial general intelligence by creating a more dynamic and context-aware conversational AI system.

## Introduction

Recent advancements in large language models (LLMs) have shown remarkable capabilities in natural language understanding and generation. However, these models often lack the ability to effectively utilize external knowledge and maintain coherent, context-aware conversations over extended periods. This research project addresses these limitations by implementing three key techniques: Self-RAG, LLM self-generation, and active memory injection.

## Key Components and Implementation

### 1. Self-RAG (Retrieval-Augmented Generation)

The system implements Self-RAG to enhance the AI's ability to access and utilize relevant information during conversations. This is achieved through the following components:

- **Vector Database**: The project uses LanceDB, a vector database, to store and efficiently retrieve conversation data and generated content.
- **Embedding Models**: The system employs embedding models (e.g., "nomic-ai/nomic-embed-text-v1.5") to convert text into vector representations for semantic search.
- **Query Mechanism**: The `ConversationModel` class implements a `query` method that searches the vector database for relevant information based on the current conversation context.

Implementation note: The `ChatTurnStrategy` class in `chat.py` orchestrates the retrieval and injection of relevant information into the conversation context.

### 2. LLM Self-Generation

The project leverages LLM self-generation to allow the AI to generate its own thoughts, reflections, and analyses. This is implemented through various pipeline types:

- **Analysis Pipeline**: Implemented in `pipeline/analyze.py`, this pipeline generates in-depth analyses of conversations, including entity recognition and reflection.
- **Journal Pipeline**: Found in `pipeline/journal.py`, this pipeline allows the AI to generate introspective journal entries, fostering a sense of continuous learning and self-awareness.
- **Chore Pipeline**: Implemented in `pipeline/chore.py`, this pipeline enables the AI to break down and execute multi-step tasks, demonstrating problem-solving capabilities.
- **Report Pipeline**: Found in `pipeline/report.py`, this pipeline generates comprehensive reports on specific topics, showcasing the AI's ability to synthesize information.

Implementation note: Each pipeline utilizes a series of prompts and LLM calls to guide the AI through a structured thought process, encouraging self-generation of content.

### 3. Active Memory Injection

The system implements active memory injection to maintain context and coherence across conversations:

- **Conscious Memory**: The `ChatTurnStrategy` class implements a `get_conscious_memory` method that retrieves and injects relevant past memories into the current conversation context.
- **Memory Weighting**: The system uses a combination of relevance scoring and temporal decay to prioritize memories, ensuring that the most pertinent information is injected into the context.

Implementation note: The `BasePipeline` class in `pipeline/base.py` includes methods like `accumulate` and `format_history` that manage the injection of memories into the conversation flow.

## Innovative Aspects

1. **Dynamic Persona Management**: The system allows for the loading and utilization of different AI personas, enabling research into how personality traits affect AGI behavior and decision-making.

2. **Flexible LLM Integration**: The project supports multiple LLM providers (OpenAI, AI Studio, Groq), allowing for comparative research across different models.

3. **Semantic Keyword Emphasis**: The system places special emphasis on capitalized words as semantically important, encouraging the AI to develop a more nuanced understanding of context and importance.

4. **Multi-Modal Pipelines**: By implementing various pipeline types (analysis, journal, chore, report), the project enables research into different aspects of AGI cognition and task execution.

## Future Directions: Improving Sparse Vectorization and Real-Time Memory

As we continue to advance our AGI research, we've identified key areas for improvement and expansion. This section outlines our plans for enhancing sparse vectorization techniques and extending our system to incorporate real-time memory management through a new 'consciousness' module.

### Improving Sparse Vectorization

Our current implementation uses basic TF-IDF vectorization for text representation. While effective, there are several paths we can explore to improve this:

1. **Enhanced Tokenization**: Implement more sophisticated tokenization techniques, such as WordPiece or SentencePiece, to better capture semantic meaning in various languages and domains.

2. **N-gram Incorporation**: Extend our vectorization to include n-grams, allowing for better capture of phrases and multi-word concepts.

3. **Named Entity Recognition (NER) Boosting**: Implement a more robust NER system and use it to boost the importance of recognized entities in our vector representations.

4. **Domain-Specific Embeddings**: Develop and incorporate domain-specific embeddings to better capture nuanced meanings in specialized fields.

5. **Hybrid Models**: Explore combining sparse vectorization with dense embeddings from transformer models to create more comprehensive text representations.

Implementation note: These improvements will primarily be implemented in the `SparseVectorizer` class in `nlp/sparse.py`. The `score_documents` method will be extended to incorporate these new techniques.

### Extension into Real-Time Memory: The Consciousness Module

To better manage the interplay between current context and long-term memory, we are developing a new `consciousness.py` module. This module will abstract and unify the management of current chat history and memory recall, providing a more cohesive and dynamic representation of the AI's "mental state."

Key features of the Consciousness module:

1. **Unified Memory Interface**: Provide a single interface for accessing both short-term (current conversation) and long-term (retrieved from database) memories.

2. **Dynamic Attention Mechanism**: Implement an attention mechanism that dynamically weights the importance of different memories based on the current context.

3. **Memory Consolidation**: Periodically consolidate short-term memories into long-term storage, mimicking human memory processes.

4. **Associative Memory Retrieval**: Implement associative memory techniques to retrieve not just directly relevant memories, but also tangentially related information that could spark creative connections.

5. **Emotional Tagging**: Incorporate an emotional tagging system for memories, allowing for more nuanced retrieval based on emotional context.

6. **Forgetting Mechanism**: Implement a controlled forgetting mechanism to prevent information overload and maintain system efficiency.

Implementation notes:
- The new `Consciousness` class will be the primary interface for both the chat system and various pipelines to interact with memory.
- It will incorporate methods from the current `ChatTurnStrategy` and `BasePipeline` classes related to memory management.
- The `get_conscious_memory` method will be expanded to include more sophisticated retrieval and filtering mechanisms.

Example structure of the `Consciousness` class:

```python
class Consciousness:
    def __init__(self, cvm: ConversationModel, config: ChatConfig):
        self.cvm = cvm
        self.config = config
        self.short_term_memory = []
        self.active_long_term_memory = []

    def update(self, new_input: str):
        # Process new input, update short-term memory

    def retrieve_relevant_memories(self, context: str) -> List[Dict]:
        # Retrieve and rank relevant memories from both short-term and long-term storage

    def consolidate_memories(self):
        # Periodically move short-term memories to long-term storage

    def forget(self):
        # Implement forgetting mechanism for less relevant or outdated memories

    def get_current_state(self) -> Dict:
        # Return the current "conscious" state, including relevant memories and context
```

This new module will allow for more sophisticated and dynamic memory management, enabling the AI to maintain a more coherent and context-aware "stream of consciousness" across interactions and tasks.

By implementing these improvements in sparse vectorization and developing the Consciousness module, we aim to significantly enhance our AGI system's ability to maintain context, make relevant associations, and exhibit more human-like memory and thought processes.