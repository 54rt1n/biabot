# assignment/models/base.py

from abc import ABC, abstractmethod
from datetime import timedelta
import lancedb
from lancedb import DBConnection
from lancedb.table  import Table
import pandas as pd
from typing import List, Tuple, Optional


class BaseModel(ABC):
    """
    The `BaseModel` class is an abstract base class that provides a common interface for models that interact with a database collection. It handles the initialization of the database connection, the embedding model, and the collection, as well as common operations such as querying, getting, and clearing the collection.
    
    Subclasses of `BaseModel` must implement the `init_collection` method to initialize the collection, and the `insert` and `query` methods to handle inserting and querying documents in the collection.
    
    The `from_uri` class method provides a convenient way to create a `BaseModel` instance from a database URI, embedding model, and collection name.
    
    The `size` property returns the total number of documents in the collection, and the `get` method allows retrieving documents based on a filter condition.
    
    The `clear` method provides a way to clear the entire collection.
    """
    def __init__(self, db: DBConnection, collection_name: str, collection: Table, sort_columns: List[str] = [], device: str = 'cpu', **kwargs):
        self.db = db
        self.collection_name = collection_name
        self.collection = collection
        self.sort_columns = sort_columns
        self.device = device

    @classmethod
    def _from_uri(cls, lancedb_uri: str, collection_name: str, device: str = 'cpu', **kwargs):
        """
        A class method that creates a new instance of the `BaseModel` class from a database URI, embedding model, and collection name.
        
        Args:
            lancedb_uri (str): The URI of the LanceDB database to connect to.
            collection_name (str): The name of the collection to use.
            device (str, optional): The device to use for the embedding model. Defaults to 'cpu'.
            **kwargs: Additional keyword arguments to pass to the `BaseModel` constructor.
        
        Returns:
            BaseModel: A new instance of the `BaseModel` class, initialized with the provided parameters.
        """
                
        db = lancedb.connect(lancedb_uri, read_consistency_interval=timedelta(0))
        collection  = cls.init_collection(db, collection_name, device)
        return cls(db=db, collection_name=collection_name, collection=collection, device=device)

    @classmethod
    @abstractmethod
    def init_collection(cls, db: DBConnection, collection_name: str, device: str = 'cpu') -> Table:
        """
        Initializes a database collection for the BaseModel class.
        
        Args:
            db (DBConnection): The database connection to use.
            collection_name (str): The name of the collection to initialize.
            device (str, optional): The device to use for the embedding model. Defaults to 'cpu'.
        
        Returns:
            Table: The initialized database collection.
        """
                
        pass
    
    @property
    def size(self) -> int:
        """
        Returns the total number of documents in the collection.

        Returns:
            int: The count of documents in the collection.
        """
        return len(self.collection)

    @abstractmethod
    def insert(self, **kwargs) -> None:
        """
        Inserts a document into the collection.
        """
        pass

    @abstractmethod
    def query(self, query_text: str, top_n: int = 5, threshold: Optional[float] = None) -> List[Tuple[str, float, str]]:
        """
        Queries the indexed documents.

        Args:
            query_text: The query as a string.
            top_n: Number of top results to return (default: 5).
            threshold: The similarity threshold for considering a document relevant.

        Returns:
            A list of documents IDs of the top-k results based on similarity, and their distances.
        """
        pass

    def get(self, filter_condition: str, limit: int = 1000) -> pd.DataFrame:
        """
        Returns documents based on a given filter condition.

        Args:
            filter_condition: A string representing the filter condition.
            limit: Maximum number of memories to return (default: 1000).

        Returns:
            A pandas DataFrame containing the filtered documents.
        """
        results = self.collection.search()\
            .where(filter_condition)\
            .limit(limit)\
            .to_pandas().sort_values(by=self.sort_columns, ascending=False)

        return results

    def clear(self):
        """
        Clears the collection.
        """
        self.db.drop_table(self.collection_name)
        self.init_collection(self.db, self.collection_name, self.device)
