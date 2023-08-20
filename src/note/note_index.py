from ..ingestion import index

class TypesenseNote():
    def __init__(self, client, index_config=None, encoder=None):
        self.client = client
        self.index = index.TypesenseIndexer(index_config, index_config['name'],
                                            data_folder=None, client=client)
        self.searcher = index.QueryIndexTypesense(vector_index='notes',
                                            fts_index='notes',
                                            client=client,
                                            embedding_column='vec',
                                            encoder=encoder,
                                            search_fields='people,places,notes,orgs,tags,dates')

    def index_note(self, note):
        self.index.index_document(note)

    def search_note(self, query):
        results = self.searcher.search(query, top_k_fts=100, top_k_vector=100)
        return results


