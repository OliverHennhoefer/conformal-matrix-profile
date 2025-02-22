from multiple_testing.sequence.saffron import DefaultSaffronGammaSequence


class BatchBY:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.alpha0: float = alpha
        self.num_test: int = 1

        self.seq = DefaultSaffronGammaSequence(gamma_exp=1.6, c=0.4374901658)
        self.r_s_plus: [float] = []
        self.r_s: [bool] = []
        self.r_total: int = 0
        self.r_sums: [float] = [0]
        self.alpha_s: [float] = []

    def test_batch(self, p_vals: list[float]) -> list[bool]:
        n_batch = len(p_vals)
        if self.num_test == 1:
            self.alpha = (
                self.alpha0  # fmt: skip
                * self.seq.calc_gamma(j=1)
            )
        else:
            self.alpha = (
                sum(
                    self.seq.calc_gamma(i) for i in range(1, self.num_test + 1)
                )
                * self.alpha0  # fmt: skip
            )
            self.alpha -= sum(
                [
                    self.alpha_s[i]
                    * self.r_s_plus[i]
                    / (self.r_s_plus[i] + self.r_sums[i + 1])
                    for i in range(0, self.num_test - 1)
                ]
            )
            self.alpha *= (n_batch + self.r_total) / n_batch

        num_reject, threshold = by(p_vals, self.alpha)

        self.r_sums.append(self.r_total)
        self.r_sums[1:self.num_test] = \
            [x + num_reject for x in self.r_sums[1:self.num_test]]  # fmt: skip
        self.r_total += num_reject
        self.alpha_s.append(self.alpha)

        r_plus = 0
        for i, p_val in enumerate(p_vals):
            p_vals[i] = 0
            r_plus = max(r_plus, by(p_vals, self.alpha)[0])
            p_vals[i] = p_val
        self.r_s_plus.append(r_plus)

        self.num_test += 1
        return [p_val <= threshold for p_val in p_vals]


def by(p_vals: [float], alpha: float) -> (int, float):
    n = len(p_vals)
    sorted_p_vals = sorted(p_vals)
    harmonic_sum = sum(1 / (i + 1) for i in range(n))

    def condition(i):
        return sorted_p_vals[i] <= alpha * (i + 1) / (n * harmonic_sum)

    left, right = 0, n
    while left < right:
        mid = (left + right) // 2
        if condition(mid):
            left = mid + 1
        else:
            right = mid

    return left, alpha * left / (n * harmonic_sum) if left else 0
