import json
import gzip
from typesense import Client
from pathlib import Path
import pandas as pd

class QueryIndexTypesense():
    def __init__(self, vector_index, fts_index, client, embedding_column, encoder, search_fields,
                 id_column='id', rrf_kl=60, rrf_kd=60):
        self.vector_index = vector_index
        self.fts_index = fts_index
        # self.data = data
        self.embedding_column = embedding_column
        self.encoder = encoder
        self.client = client
        self.search_fields = search_fields
        self.id_column = id_column
        self.rrf_kl = rrf_kl
        self.rrf_kd = rrf_kd

    def format_fts_response(self, docs, columns=None):
        search_ids = dict()
        search_results = dict()

        print(f'number of fts docs: {len(docs)}')
        for position, doc in enumerate(docs, start=1):
            id = doc['document']['id']
            search_ids[id] = position
            search_results[id] = doc['document']

        return search_ids, search_results

    def format_vector_response(self, docs, columns=None):
        search_ids = dict()
        search_results = dict()
        print(f'number of vector docs: {len(docs)}')
        for position, doc in enumerate(docs, start=1):
            id = doc['document']['id']
            search_ids[id] = position
            search_results[id] = doc['document']

        return search_ids, search_results


    def search_bm25(self, query, fields=None, include_fields=None, top_k=10):
        if fields is None:
            fields = self.search_fields
        fts_query = {
            'q'         : query,
            'query_by'  : fields,
            'per_page'  : top_k
        }
        if include_fields is not None:
            fts_query['include_fields'] = include_fields

        self.bm25_results = self.client.collections[f'{self.fts_index}'].documents.search(fts_query)
        search_ids, search_results = self.format_fts_response(self.bm25_results['hits'])
        return search_ids, search_results

    def search_vector(self, query, top_k=10):
        query_vec = ','.join([str(x) for x in self.encoder.encode(query)])

        search_requests = {
            'searches': [
                {
                    'collection': f'{self.vector_index}',
                    'q' : '*',
                    'vector_query': f'{self.embedding_column}:([{query_vec}], k:{top_k})',
                    'per_page'  : top_k
                }
            ]
        }

        # Search parameters that are common to all searches go here
        common_search_params =  {}
        self.vec_results = self.client.multi_search.perform(search_requests, common_search_params)['results'][0]['hits']
        search_ids, search_results = self.format_vector_response(self.vec_results)
        return search_ids, search_results

    def find_overlap(self, vector_ids, fts_ids):
        overlap = set(vector_ids.keys()).intersection(set(fts_ids.keys()))
        print(overlap)
        vector_overlap = {id:position for id, position in vector_ids.items() if id in overlap}
        fts_overlap = {id:position for id, position in fts_ids.items() if id in overlap}
        return vector_overlap, fts_overlap

    def rrf_reranking(self, vector_overlap, fts_overlap, fts_weight=1, vector_weight=1):
        overlap_ids = list(vector_overlap.keys())
        hybrid_rank = dict()
        for id in overlap_ids:
            vector_position = 1 / (self.rrf_kd + vector_overlap[id])
            fts_position = 1 / (self.rrf_kl + fts_overlap[id])
            hybrid_rank[id] = vector_position * vector_weight + fts_position * fts_weight

        hybrid_rank = [(k,v) for k,v in hybrid_rank.items()]
        hybrid_rank = sorted(hybrid_rank, key=lambda x: x[1])[::-1]
        return hybrid_rank

    def merge_parallel_indexes(self, vector_results, fts_results):
        vector_overlap, fts_overlap = self.find_overlap(vector_results[0], fts_results[0])
        self.reranking_idx = self.rrf_reranking(vector_overlap, fts_overlap)
        overlap_data = [vector_results[1][x[0]] for x in self.reranking_idx]
        return overlap_data


    def search(self, query, top_k_vector=100, top_k_fts=100):
        vector_results = self.search_vector(query, top_k=top_k_vector)
        fts_results = self.search_bm25(query, top_k=top_k_fts)

        print(len(vector_results))
        print(len(fts_results))

        self.overlap_results = self.merge_parallel_indexes(vector_results, fts_results)
        return self.overlap_results

    def single_doc_retrieval(self, docid):
        doc = self.client.collections[self.fts_index].documents[f'{docid}'].retrieve()
        return doc

class TypesenseIndexer():
    def __init__(self, index_config, index_name, data_folder, client):
        self.index_config = index_config
        self.data_folder = [x for x in Path(data_folder).glob('*.*')]
        self.client = client
        self.index_name = index_name
        self._check_indexes()
    def _check_indexes(self):
        collections = self.client.collections.retrieve()
        collections = [collection['name'] for collection in collections]
        if self.index_config['name'] not in collections:
            self._create_index()

    def _create_index(self):
        self.client.collections.create(self.index_config)
        return True

    def load_data(self, documents):
        self.client.collections[self.index_name].documents.import_(documents, {'action': 'create'})

    def index_documents(self, documents):
        for file in self.data_folder:
            if file.suffix == '.json':
                with open(file,'r') as f:
                    documents = json.load(f)
                    print(file)
                    print(len(documents))
                    # documents = [x.strip('\n') for x in f.readlines()]
                    self.load_data(documents)

            elif file.suffix == '.csv':
                documents = pd.read_csv(file)
                documents = documents.to_dict(orient='records')
                self.load_data(documents)

    def index_document(self, document):
        self.client.collections[self.index_config['name']].documents.create(document)