# biabot/models/conversation.py

from datetime import datetime
from lancedb import DBConnection
from lancedb.embeddings import get_registry, SentenceTransformerEmbeddings
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import Table
import numpy as np
import pandas as pd
import time
from typing import Optional, Set, List
from wonderwords import RandomWord

from ..constants import DOC_ANALYSIS, DOC_CONVERSATION, DOC_JOURNAL, DOC_NER, DOC_STEP, LISTENER_ALL
from ..nlp.sparser import SparseVectorizer
from .base import BaseModel

COLUMNS = ['doc_id', 'document_type', 'user_id', 'persona_id', 'conversation_id', 'role', 'content', 'branch', 'sequence_no']


class ConversationModel(BaseModel):
    collection_name : str = 'conversation'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vectorizer = SparseVectorizer()

    @classmethod
    def from_uri(cls, lancedb_uri: str, device: str = 'cpu', **kwargs):
        return super()._from_uri(lancedb_uri, cls.collection_name, device, **kwargs)
    
    @classmethod
    def init_collection(cls, db: DBConnection, collection_name: str = "conversation", device: str = 'cpu') -> Table:
        """
        Initializes a conversation collection in the database with a specified embedding model.
        
        Args:
            db (DBConnection): The database connection to use.
            collection_name (str): The name of the collection to initialize. Defaults to 'conversation'.
            device (str, optional): The device to use for the embedding model, defaults to 'cpu'.
        
        Returns:
            Table: The initialized conversation collection.
        """
                
        st_registry : SentenceTransformerEmbeddings = get_registry().get("sentence-transformers")
        # NOTE: This is required configuration for stella_en_400M_v5 to work on CPU. Stella is an excellent model.
        embedding_model = "dunzhang/stella_en_400M_v5"
        model = st_registry.create(name=embedding_model, device=device, allow_remote_code=True, config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False})

        class ConversationEntry(LanceModel):
            """
            A document with an id and content. The vector is duck-typed to the embedding model's vector type, and the content mapped to the model input.
            """
            index: Vector(model.ndims()) = model.VectorField(prompt_name="s2p_query")
            
            # Our primary key
            doc_id: str
            document_type: str
            
            # Our natural key
            user_id: str
            persona_id: str
            conversation_id: str
            branch: int
            sequence_no: int
            
            # Conversation metadata
            listener_id: str # The id of the listener, or all, self, etc.
            reference_id: str # If we are referencing another document or conversation
            
            # The conversation turn
            role: str
            content: str = model.SourceField()
            
            # We also need to be able to up/downweight the conversation turn
            weight: float

            timestamp: int
            inference_model: str # The model that generated the conversation turn
            status: int

        collection = db.create_table(collection_name, schema=ConversationEntry, exist_ok=True)
        return collection
    
    def insert(self, conversation_id: str, sequence_no: int, role: str, content: str, document_type: str = "conversation",
               doc_id: Optional[str] = None, reference_id: Optional[str] = None,
               user_id: str = 'user', persona_id: str = 'assistant', listener_id: str = LISTENER_ALL,
               branch: int = 0, inference_model: Optional[str] = None, weight: float = 1.0, timestamp: int = int(time.time()),
               status: int = 0, **kwargs) -> None:
        """
        Inserts a conversation in to the collection.
        """

        if doc_id is None:
            doc_id = self.next_doc_id()

        if reference_id is None:
            reference_id = conversation_id

        print(f"Inserting {doc_id} into {self.collection.name}")

        if inference_model is None:
            inference_model = "default"
            
        # Add the document and its embedding to the collection
        data = {
            "doc_id": doc_id,
            "document_type": document_type,
            "user_id": user_id,
            "persona_id": persona_id,
            "conversation_id": conversation_id,
            "branch": branch,
            "sequence_no": sequence_no,
            "listener_id": listener_id,
            "reference_id": reference_id,
            "role": role,
            "content": content,
            "weight": weight,
            "timestamp": timestamp,
            "inference_model": inference_model,
            "status": status,
        }
        self.collection.add(data=[data])
    
    def train_vectorizer(self, documents: List[str]) -> None:
        """
        Trains the vectorizer on a list of documents.
        """
        self.vectorizer.clear()
        self.vectorizer.fit(documents)

    def query(self, query_texts: List[str], filter_doc_ids: Optional[Set[str]] = None, filter_text: Optional[str] = None, top_n: Optional[int] = None, document_type: Optional[str] = None, turn_decay: float = 0.7, temporal_decay: float = 0.7, filter_metadocs: bool = True) -> pd.DataFrame:
        """
        Queries the conversation collection and returns a DataFrame containing the top `top_n` most relevant conversation entries based on the given query texts, filters, and decay factors.
        
        The query is performed using the collection's search functionality, with optional filters applied to exclude certain document types, document IDs, and text content. The relevance score for each entry is calculated as a combination of the text similarity to the query texts, the temporal decay based on the entry's timestamp, and the entry's weight.
        
        The returned DataFrame includes the following columns:
        - `content`: The content of the conversation entry.
        - `timestamp`: The timestamp of the conversation entry.
        - `weight`: The weight of the conversation entry.
        - `role`: The role of the speaker (either 'user' or 'assistant').
        - `user_id`: The ID of the user who made the conversation entry.
        - `persona_id`: The ID of the persona associated with the conversation entry.
        - `document_type`: The type of the document.
        - `date`: The date and time of the conversation entry.
        - `speaker`: The speaker of the conversation entry (either the user's ID or the persona's ID).
        - `score`: The relevance score of the conversation entry.
        """
                
        qb = self.collection.search('\n'.join(query_texts))
        wheres = []
        if filter_metadocs:
            wheres.append(f"document_type != '{DOC_NER}' and document_type != '{DOC_STEP}'")
        if filter_doc_ids:
            for doc_id in filter_doc_ids:
                wheres.append(f"doc_id != '{doc_id}'")
        if filter_text:
            wheres.append(f"({filter_text})")
        if document_type:
            wheres.append(f"document_type = '{document_type}'")
        if len(wheres) > 0:
            qb = qb.where(' and '.join(wheres), prefilter=True)

        # lancedb's limit doesn't really work, so we have to do it ourselves
        results : pd.DataFrame = qb.limit(1000).to_pandas()
        if results.empty:
            results['date'] = ''
            results['speaker'] = ''
            results['score'] = 0.0
            return results[COLUMNS + ['date', 'speaker', 'score']]

        results['dscore'] = 1 / results['_distance']

        if len(query_texts) > 0:
            self.train_vectorizer(results['content'].tolist())
            sim_matrix = self.vectorizer.query(queries=query_texts)
            print(f"Sim Matrix shape: {sim_matrix.shape}")

            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            for n in range(sim_matrix.shape[0]):
                letters = ""
                rem = n
                while rem > 0:
                    letter = alphabet[rem % 26]
                    rem = rem // 26
                    letters = letter + letters
                results[f'sim_{letters}'] = sim_matrix[n]
                # If zero is our most recent at 1/1, then we want our decay to follow our decay rate
                results[f'decay_{letters}'] = turn_decay**n
                results[f'score_{letters}'] = sim_matrix[n] * turn_decay**n
        else:
            results['score_all'] = 1.0

        results['length'] = np.log2(results['content'].str.len())
        # Scale with the following rules - 
        # A slope of 7 days, and a y-intercept of 1
        # Beta binomial curve optimizing for recency

        current_time = int(time.time())
        seven_days_in_seconds = 7 * 24 * 60 * 60
    
        # Calculate decay factor
        results['temporal_decay'] = np.exp(-temporal_decay * (current_time - results['timestamp']) / seven_days_in_seconds)
     
        # Now, we sum the scores, and multiply by the dscore
        results['score'] = results.filter(regex='score_').max(axis=1) * (results['dscore'] / 2.0 + 0.5) * results['weight'] * (results['length'] / 2.0 + 0.5) * results['temporal_decay']

        results['date'] = results['timestamp'].apply(lambda d: datetime.fromtimestamp(d).strftime('%Y-%m-%d %H:%M:%S'))

        # if the role is user, we use the user_id as the speaker, if the role is assistant, we use the persona_id
        results['speaker'] = results.apply(lambda row: row['user_id'] if row['role'] == 'user' else row['persona_id'], axis=1)

        # return our filtered results
        return results.sort_values(by='score', ascending=False).head(top_n)[COLUMNS + ['date', 'speaker', 'score']]

    def get_conscious(self, persona_id: str, top_n: int) -> pd.DataFrame:
        """
        Returns a DataFrame containing the top `top_n` most "conscious" journal entries for the given `persona_id`.
        
        The "consciousness" score is calculated as a stochastic-recency score, where each entry is assigned a random score between 0 and 1, which is then multiplied by the entry's `weight` to produce the final score.
        
        The returned DataFrame includes the following columns:
        - `content`: The content of the journal entry.
        - `timestamp`: The timestamp of the journal entry.
        - `weight`: The weight of the journal entry.
        - `role`: The role of the speaker (either 'user' or 'assistant').
        - `user_id`: The ID of the user who made the journal entry.
        - `persona_id`: The ID of the persona associated with the journal entry.
        - `document_type`: The type of the document (for now, always `DOC_JOURNAL`).
        - `date`: The date and time of the journal entry.
        - `speaker`: The speaker of the journal entry (either the user's ID or the persona's ID).
        """

        results : pd.DataFrame = self.collection.search().where(f"document_type = '{DOC_JOURNAL}' and persona_id = '{persona_id}'").limit(-1).to_pandas()

        # our score will be stochastic, to bring in a variety of entries
        results['score'] = np.random.rand(len(results)) * results['weight']

        results['date'] = results['timestamp'].apply(lambda d: datetime.fromtimestamp(d).strftime('%Y-%m-%d %H:%M:%S'))
        results['speaker'] = results.apply(lambda row: row['user_id'] if row['role'] == 'user' else row['persona_id'], axis=1)

        return results.sort_values(by='score', ascending=False).reset_index(drop=True)[COLUMNS + ['date', 'speaker']].head(top_n)

    def next_conversation_id(self, user_id: str, persona_id: str) -> str:
        """
        Returns a valid unique conversation ID for a given user.
        """

        # A converation id is three random words separated by dashes
        random_words = RandomWord()
        conversation_id = "-".join(random_words.word() for _ in range(2))

        # Check if the user already has this conversation id
        exists : pd.DataFrame = self.collection.search().where(
            f"user_id = '{user_id}' and persona_id = '{persona_id}'and conversation_id = '{conversation_id}'"
            ).limit(1).to_pandas()

        # If it does, generate a new one, recursively
        if not exists.empty:
            return self.next_conversation_id(user_id)

        return conversation_id

    def next_doc_id(self) -> str:
        """
        Returns a valid unique doc ID.
        """

        # A converation id is three random words separated by dashes
        random_words = RandomWord()
        doc_id = "-".join(random_words.word() for _ in range(3))

        # Check if the user already has this conversation id
        exists : pd.DataFrame = self.collection.search().where(
            f"doc_id = '{doc_id}'"
            ).limit(1).to_pandas()

        # If it does, generate a new one, recursively
        if not exists.empty:
            return self.next_doc_id()

        return doc_id

    def get_conversation_history(self, conversation_id: str, query_text: Optional[str] = None, filter_text: Optional[str] = None, top_n: int = -1,
                                 **kwargs) -> pd.DataFrame:
        """
        Returns the conversation history for a given user and conversation ID.
        """
        if query_text is not None and top_n <= 0:
            top_n = 100
        filter_text = f"conversation_id = '{conversation_id}'" if filter_text is None else f"conversation_id = '{conversation_id}' and ({filter_text})"
        results : pd.DataFrame = (self.collection.search() if query_text is None else self.collection.search(query_text))\
            .where(filter_text, prefilter=True)\
            .limit(top_n)\
            .to_pandas().sort_values(by=['timestamp', 'branch', 'sequence_no'], ascending=True)

        # Now we need to make the sequence numbers unique by filtering out all branches except for the highest
        # First we calculate the max branch for each sequence number
        max_seq = results.groupby('sequence_no')['branch'].transform('max')
        # Now we mask all of the rows that are not the max branch, but first we have to join our data back
        results = results.join(max_seq, on='sequence_no', rsuffix='_max')
        # Now we mask all of the rows that are not the max branch
        results = results[results['branch'] == results['branch_max']]

        results['date'] = results['timestamp'].apply(lambda d: datetime.fromtimestamp(d).strftime('%Y-%m-%d %H:%M:%S'))
        results['speaker'] = results.apply(lambda row: row['user_id'] if row['role'] == 'user' else row['persona_id'], axis=1)

        return results[COLUMNS + ['date', 'speaker']]

    def ner_query(self, query_text: str, filter_text: Optional[str] = None, top_n: int = -1, **kwargs) -> pd.DataFrame:
        """
        Returns the convesation id's for a given query.
        """
        filter_text = f"document_type = '{DOC_NER}'" if filter_text is None else f"document_type = '{DOC_NER}' and ({filter_text})"
        return self.query([query_text], filter_text, top_n, **kwargs)
    
    def get_next_branch(self, document_type: str, user_id: str, persona_id: str, conversation_id: str) -> int:
        """
        Returns the maximum branch number for a given user and conversation ID.
        """
        results : pd.DataFrame = self.collection.search().where(
            f"document_type = '{document_type}' and user_id = '{user_id}' and persona_id = '{persona_id}' and conversation_id = '{conversation_id}'"
            ).to_pandas()

        if results.empty:
            print("No branches found")
            return 0
        
        next_branch = results['branch'].max() + 1
        print(f"Next branch: {next_branch}")
        return next_branch

    def get_conversation_report(self):
        all_df = self.collection.to_pandas()
        # determine conversations without analysis:
        # we need to find all conversation ids by document type
        docs = all_df.groupby(['document_type', 'conversation_id']).size().reset_index()
        # reshape so docuemnt types are columns
        docs = docs.pivot(index='conversation_id', columns='document_type', values=0).fillna(0).reset_index()
        docs
        conversation_time = all_df.groupby('conversation_id').agg({'timestamp': 'max'}).reset_index()
        conversation_time.columns = ['conversation_id', 'timestamp_max']

        conversation_report = pd.merge(docs, conversation_time, on='conversation_id').sort_values('timestamp_max')
        return conversation_report

    @property
    def next_analysis(self) -> Optional[str]:
        cr = self.get_conversation_report()
        conversation_mask = cr[DOC_CONVERSATION] > 0
        if DOC_ANALYSIS not in cr.columns:
            next_analysis = cr[conversation_mask].sort_values(by='timestamp_max', ascending=True).head(1)
        else:
            analysis_mask = cr[DOC_ANALYSIS] == 0
            next_analysis = cr[conversation_mask & analysis_mask].sort_values(by='timestamp_max', ascending=True).head(1)
        if next_analysis.empty:
            return None
        else:
            return next_analysis['conversation_id'].values[0]