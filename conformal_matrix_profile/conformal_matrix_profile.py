import numpy as np
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
        pieces: int = 2**6,
    ):
        self.subseq_len: int = subseq_len
        self.window_len: int = window_len
        self.calib_size: int = calib_size

        self.pieces: int = pieces
        self.warmed_up: bool = False

        self.search_window: deque = deque(maxlen=self.window_len)
        self.l_matrix_prof: deque = deque(maxlen=self.calib_size)

    def learn_one(self, instance) -> None:
        self.search_window.append(instance)
        self.warmed_up = len(self.search_window) == self.search_window.maxlen

    def estimate_one(self, instance) -> float:
        if self.warmed_up:
            p_val = -1
            min_dist = self._get_min_dist_profile(instance)
            if len(self.l_matrix_prof) == self.calib_size:
                p_val = self._compute_p_val(min_dist)
            self.l_matrix_prof.append(min_dist)
            return p_val
        return -1

    def _get_min_dist_profile(self, instance):
        q = list(self.search_window)[-(self.subseq_len - 1) :]
        q.append(instance)
        t = list(self.search_window)[: -(self.subseq_len - 1)]
        distance_profile = mass_approx(t, q, pieces=self.pieces)
        return min(distance_profile)

    def _compute_p_val(self, min_dist):
        sum_smaller = sum(self.l_matrix_prof >= min_dist)
        if sum_smaller == 0:
            data = pd.Series(self.l_matrix_prof)
            data = data.apply(lambda x: x.real)
            threshold = data.nlargest(50).iloc[-1]  # 30 - 50
            exceed = data[data >= threshold]
            params = genpareto.fit(exceed - threshold, floc=0)
            c, loc, scale = params
            return genpareto.pdf((min_dist - threshold).real, c=c, loc=loc, scale=scale)
        return (1.0 + sum_smaller) / (1.0 + len(self.l_matrix_prof))
