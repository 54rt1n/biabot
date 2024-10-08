# BIABot: Advanced Chatbot with Active Memory

BIABot is a sophisticated chatbot system featuring active memory, various conversation pipelines, and integration with large language models. It's designed for complex interactions, data analysis, and task management.

The main objectives are:

1. To create a flexible and extensible chatbot framework
2. To implement a system for maintaining and querying conversation history
3. To develop various analysis pipelines for processing and learning from conversations
4. To showcase the integration of large language models (LLMs) with vector databases for efficient information retrieval

## Features

- Conversation management with persistent storage
- Multiple conversation pipelines (analysis, journal, chore, report)
- Integration with large language models (LLMs) like GPT
- Active memory system for context-aware responses
- Document management and search capabilities
- Flexible configuration options

## Limitations

- Does not work on Windows: I was unable to install lancedb on Windows, which is why it wasn't used for assignment 5...

## Project Structure

The project is organized as a Python package named `biabot` with the following key components:

- `__main__.py`: Entry point for the CLI application
- `chat.py`: Implements the core chat functionality
- `config.py`: Manages configuration settings
- `constants.py`: Defines constant values used throughout the project
- `documents.py`: Handles document management
- `llm.py`: Provides an interface for interacting with language models
- `patterns.py`: Contains regex patterns for text processing
- `persona.py`: Defines the AI assistant's personality

### Models
- `conversation.py`: Manages conversation data storage and retrieval

### Pipelines
- `analyze.py`: Performs conversation analysis
- `chore.py`: Handles task-oriented conversations
- `journal.py`: Generates reflective journal entries
- `report.py`: Creates comprehensive reports

### NLP
- `sparse.py`: Implements sparse vector with TF-IDF
- `sparser.py`: Implements a BPE sparse vectorizer with BM25

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/54rt1n/biabot.git
   cd biabot
   ```

2. Install dependencies:
   ```
   poetry install
   ```

3. Set up environment variables:
   Copy the `.env.example` file to `.env` and fill in the required values:
   ```
   cp sample.env .env
   ```

## Usage Examples

Refer to the `__main__.py` file for a complete list of available commands. Here are some key examples:

```bash
# Show help
python -m biabot

# Start a chat session
python -m biabot chat --user-id Student --persona-id Professor --conversation-id chat-001
#Note: Active memory does not include the current chat session.

# Run analysis pipeline
python -m biabot pipeline-analysis Student Professor chat-001 --mood curious What are the key takeaways from our conversation?

# Generate a journal entry
python -m biabot pipeline-journal Professor journal-001 Reflect on recent conversations

# Generate a report
python -m biabot pipeline-report Professor report-001 --document AGI.md Summarize your understanding of AGI

# Process a chore
python -m biabot pipeline-chore Professor chore-001 Organize project tasks

```

### Import and and Exporting your history

```bash
#To export a conversation to a JSONL file:
python -m biabot export-all

#To import a conversation from a JSONL file:
python -m biabot import-dump ./export/professor-001.jsonl
```

### Listing Conversations

```bash
#To list all conversations in the database:
python -m biabot list-conversations

# To view a matrix of all conversations and their doc types:
python -m biabot matrix
```

## LLM Providers

To use `groq`, you will need to `pip install groq`.

To use `ai_studio`, you will need to `pip install google-generativeai`.

The `openai` provider is the default, and can be used both with the OpenAI API and many local models and alternative providers.

To configure `openai` provider, set the following environment variables:
```bash
MODEL_URL=http://127.0.0.1:5000/v1
MODEL_NAME=Mistral-Small-22B.Q5_K_M.gguf
```

This example is for a local model running on port 5000, being served by llama.cpp.

## License

This project is licensed under the MIT License.

## Acknowledgments

- This project was developed as part of the BIA6304 course.
- Thank you to Claude Sonnet 3.5 and GPT-4o for their contributions to the project.
- Special thanks to the local llm community for their work on development of cutting-edge AI models and tools.

## Project Requirements

Students should use this sample implementation as a reference to understand the project requirements:

1. Implement a conversational AI system
2. Integrate with a vector database for efficient information retrieval
3. Implement an interface for use

## Evaluation Criteria

Your projects will be evaluated based on:

1. Functionality
2. Proper use of vector databases and language models
3. Proper documentation and code comments
