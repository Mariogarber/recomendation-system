import numpy as np

class NDCG:
    def __init__(self, k):
        self.k = k

    def dcg(self, relevance_scores):
        return sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores[:self.k])])

    def idcg(self, relevance_scores):
        sorted_relevance = sorted(relevance_scores, reverse=True)
        return self.dcg(sorted_relevance)

    def ndcg(self, relevance_scores):
        ideal_dcg = self.idcg(relevance_scores)
        if ideal_dcg == 0:
            return 0.0
        return self.dcg(relevance_scores) / ideal_dcg