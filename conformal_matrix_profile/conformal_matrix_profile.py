from statistics import mean

import pandas as pd

from scipy.stats import genpareto
from collections import deque

from conformal_matrix_profile.similarity_search.mass_approx import mass_approx


class OnlineConformalMatrixProfile:
    def __init__(
        self,
        subseq_len: int,
        window_len: int,
        calib_size: int,
        tail_frac: float,
        pieces: int = 2**6,
    ):
        self.subseq_len: int = subseq_len
        self.window_len: int = window_len  # search window
        self.calib_size: int = calib_size

        self.tail_fac: float = tail_frac
        self.pieces: int = pieces

        self.warmed_up: bool = False
        self.search_window: deque = deque(maxlen=self.window_len)
        self.l_matrix_prof: deque = deque(maxlen=self.calib_size)

    def learn_one(self, instance) -> None:
        self.search_window.append(instance)
        self.warmed_up = len(self.search_window) == self.search_window.maxlen

    def estimate_one(self, instance) -> (float, float):
        if self.warmed_up:
            p_val = -1
            min_dist = self._get_min_dist_profile(instance)
            if len(self.l_matrix_prof) == self.calib_size:
                p_val = self._compute_p_val(min_dist)
            self.l_matrix_prof.append(min_dist)
            return p_val, min_dist
        return -1, -1

    def _get_min_dist_profile(self, instance):
        q = list(self.search_window)[-(self.subseq_len - 1) :]
        q.append(instance)
        t = list(self.search_window)[: -(self.subseq_len - 1)]
        distance_profile = mass_approx(t, q, pieces=self.pieces)
        return min(distance_profile.real)

    def _compute_p_val(self, min_dist) -> float:
        sum_smaller = sum(self.l_matrix_prof >= min_dist)
        if sum_smaller == 0:
            data = pd.Series(self.l_matrix_prof)
            frac = int(len(self.l_matrix_prof) * self.tail_fac)
            threshold = data.nlargest(frac).iloc[-1]
            exceed = data[data >= threshold]
            c, loc, scale = genpareto.fit(exceed - threshold, floc=0)
            covered = 1.0 / (1.0 + len(self.l_matrix_prof))
            pareto_p = genpareto.pdf(
                (min_dist - threshold).real, c=c, loc=loc, scale=scale
            )
            return covered * pareto_p
        return (1.0 + sum_smaller) / (1.0 + len(self.l_matrix_prof))

    """def unlearn(self, decisions: [bool]):
        discovery_ids = [i for i, value in enumerate(decisions) if value]  # which
        homogenized_ids = [len(self.l_matrix_prof) - len(decisions) + i for i in discovery_ids]
        for i, id in enumerate(homogenized_ids):
            del self.l_matrix_prof[id]
            self.l_matrix_prof.insert(id, mean(self.l_matrix_prof))"""
