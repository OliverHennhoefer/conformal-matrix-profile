class DefaultSaffronGammaSequence:
    """Proposed default gamma sequence for SAFFRON[1]_.

    [1] Ramdas, A., T. Zrnic, M. J. Wainwright, and M. I. Jordan.
    SAFFRON: an adaptive algorithm for online control of the FDR.
    In Proceedings of the Internat. Conference on Machine Learning, 2018."""

    def __init__(self, gamma_exp, c):
        self.gamma_exp = gamma_exp
        self.c = c

    def calc_gamma(self, j: int):
        return j**self.gamma_exp if self.c is None else self.c / j**self.gamma_exp
