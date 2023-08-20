import numpy as np
import pandas as pd
def opa_rank(subjective_rank, objective_rank,
             a=0.5, b=0.5, c=1, a_e=1, b_e=1, c_e=1,
             a_k=20, b_k=20):
    return a*(1/(a_k + subjective_rank))**a_e + \
           b*(1/(b_k + objective_rank))**b_e + \
           c*((1/(a_k + subjective_rank))*(1/(b_k + objective_rank)))**c_e

class HybridRerank():
    def __init__(self, index,
                 a=0.5, b=0.5, c=1, a_e=1, b_e=1, c_e=1,a_k=20, b_k=20):
        self.index = index
        self.a = a
        self.b = b
        self.c = c
        self.a_e = a_e
        self.b_e = b_e
        self.c_e = c_e
        self.a_k = a_k
        self.b_k = b_k
        self.opa_params = {'a':a, 'b':b, 'c':c, 'a_e':a_e, 'b_e':b_e, 'c_e':c_e,
                           'a_k':a_k, 'b_k':b_k}

    def dense_rank(self, results):
        # Apply ranks to the dense score
        dense_scores = [record['score'] for record in results]
        dense_scores = np.argsort(dense_scores)
        for record, rank in zip(results, dense_scores):
            record['dense_rank'] = rank
        return results

    def sparse_rerank(self, results):
        # return sorted results
        results = self.dense_rank(results)
        for record in results:
            record['opa_rank'] = self._opa_rank(record['score'],record['dense_rank'])
        results = sorted(results, key=lambda x: x['opa_rank'])
        return results

    def _opa_rank(self, subjective_col, objective_col):
        # modified reranking formula
        return opa_rank(subjective_col, objective_col, **self.opa_params)

class HybridSearch():
    def __init__(self, index, top_n=10, join_method='inner',
                 a=0.5, b=0.5, c=1, a_e=1, b_e=1, c_e=1,a_k=20, b_k=20):
        self.index = index
        self.a = a
        self.b = b
        self.c = c
        self.a_e = a_e
        self.b_e = b_e
        self.c_e = c_e
        self.a_k = a_k
        self.b_k = b_k
        self.opa_params = {'a':a, 'b':b, 'c':c, 'a_e':a_e, 'b_e':b_e, 'c_e':c_e,
                           'a_k':a_k, 'b_k':b_k}
        self.top_n = top_n
        self.join_method = join_method

    def sparse_results(self):
        pass

    def dense_results(self):
        # Apply ranks to the dense score
        # keep top k results for merging
        pass

    def merge_results(self, sparse_results, dense_results):
        # Join results on key
        if self.join_method == 'inner':
            # inner join on ids
            results = pd.merge(sparse_results, dense_results, how='inner')
        else:
            # outer join on ids
            results = pd.merge(sparse_results, dense_results, how='outer')

        for record in results:
            record['opa_rank'] = self._opa_rank(record['score'],record['dense_rank'])
        results = sorted(results, key=lambda x: x['opa_rank'])
        return results

    def _opa_rank(self, subjective_col, objective_col):
        # modified reranking formula
        return opa_rank(subjective_col, objective_col, **self.opa_params)

    def return_results(self):
        pass