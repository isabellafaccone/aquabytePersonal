import numbers
import numpy as np

class Normalizer:
    def __init__(self, normalize_params):
        self.normalize_params = normalize_params
        if normalize_params:
            self._check_score_normalize_params(normalize_params)


    def _check_score_normalize_params(self, params):
        assert 'overall' in params.keys(), 'missing overall key'
        for pen in params.keys():
            cutoffs = [el[0] for el in params[pen]]
            accept_rates = [el[1] for el in params[pen]]

            assert min(cutoffs) == 0.0, 'min not 0'
            for cutoff, accept_rate in zip(cutoffs, accept_rates):
                assert isinstance(cutoff, numbers.Number), 'cutoff wrong type' + str(type(cutoff))
                assert isinstance(accept_rate, numbers.Number), 'accept rate wrong type' + str(type(accept_rate))


    def _normalizeScore(self, cutoffs2acceptrates: dict, quality_score: float) -> float:
        """Maps raw quality score to normalize perpen quality score
        normalize_params is with all seen pen ids for keys, each mapping to a dict
        that itself maps score cutoff to empirical accept rate. We find the closest score
        cutoff below the quality score, and re-assign the quaity score to the empirical quality_score
        we get. If our penId is not in normalize_params, we use the overall key"""
        cutoffs = [el[0] for el in cutoffs2acceptrates]
        rates = [el[1] for el in cutoffs2acceptrates]
        return np.interp(quality_score, cutoffs, rates)


    def getNormalizedScore(self, score: float, pen_id: int) -> float:
        if self.normalize_params:
            cutoffs2acceptrates = self.normalize_params.get(str(pen_id), None) or self.normalize_params['overall']
            n_score = self._normalizeScore(cutoffs2acceptrates, score)
            return n_score
        else:
            return score
