import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

class JSONLoader():
    def __int__(self, folder: str, text_columns: list, index,
                ending='csv', batch_size=10, vectorizer=None):
        self.folder = Path(folder)
        self.files = sorted(list(self.folder.glob(f'*.{ending}')))
        self.vectorizer = vectorizer if vectorizer is not None else SentenceTransformer('all-MiniLM-L6-v2')
        self.text_columns = text_columns
        self.batch_size = batch_size
        self.index = index

    def batch_load(self):
        for file in self.files:
            with open(file, 'r') as f:
                data = json.load(f)

            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size]
                for col in self.text_columns:
                    vecs = self.vectorizer.encode([record[col] for record in batch])
                    for record, vec in zip(batch, vecs):
                        batch[f'{col}_vec'] = vec

                self.index.index_documents(batch)


class CSVLoader():
    def __int__(self, folder: str, text_columns: list, index,
                ending='csv', batch_size=10, vectorizer=None):
        self.folder = Path(folder)
        self.files = sorted(list(self.folder.glob(f'*.{ending}')))
        self.vectorizer = vectorizer if vectorizer is not None else SentenceTransformer('all-MiniLM-L6-v2')
        self.text_columns = text_columns
        self.batch_size = batch_size
        self.index = index

    def batch_load(self):
        for file in self.files:
            data = pd.read_csv(file)

            for col in self.text_columns:
                data[col] = data[col].fillna('')

            data = data.to_dict(orient='records')

            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size]
                for col in self.text_columns:
                    vecs = self.vectorizer.encode([record[col] for record in batch])
                    for record, vec in zip(batch, vecs):
                        batch[f'{col}_vec'] = vec

                self.index.index_documents(batch)

class ParquetLoader():
    def __int__(self, folder: str, text_columns: list, index,
                ending='csv', batch_size=10, vectorizer=None):
        self.folder = Path(folder)
        self.files = sorted(list(self.folder.glob(f'*.{ending}')))
        self.vectorizer = vectorizer if vectorizer is not None else SentenceTransformer('all-MiniLM-L6-v2')
        self.text_columns = text_columns
        self.batch_size = batch_size
        self.index = index

    def batch_load(self):
        for file in self.files:
            data = pd.read_parquet(file)

            for col in self.text_columns:
                data[col] = data[col].fillna('')

            data = data.to_dict(orient='records')

            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size]
                for col in self.text_columns:
                    vecs = self.vectorizer.encode([record[col] for record in batch])
                    for record, vec in zip(batch, vecs):
                        batch[f'{col}_vec'] = vec

                self.index.index_documents(batch)


class SQLLoader():
    def __init__(self):
        pass

