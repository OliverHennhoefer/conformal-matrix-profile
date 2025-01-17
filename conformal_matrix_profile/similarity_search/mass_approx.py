import numpy as np

import conformal_matrix_profile.similarity_search.mass as mass_utils


def mass_approx(ts, query, pieces):
    """
    Compute the distance profile for the given query over the given time
    series. This version of MASS is hardware efficient given the right number
    of pieces.

    Parameters
    ----------
    ts : array_like
        The array to create a rolling window on.
    query : array_like
        The query.
    pieces : int
        Number of pieces to process. This is best as a power of 2.

    Returns
    -------
    An array of distances.

    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.
        If pieces is less than the length of the query.
    """
    ts, query = mass_utils.check_series_and_query(ts, query)

    m = len(query)

    if pieces < m:
        raise ValueError("pieces should be larger than the query length.")

    n = len(ts)
    k = pieces
    x = ts
    dist = np.array([])

    # compute stats in O(n)
    meany = np.mean(query)
    sigmay = np.std(query)

    meanx = mass_utils.moving_average(x, m)
    meanx = np.append(np.ones([1, len(x) - len(meanx)]), meanx)
    sigmax = mass_utils.moving_std(x, m)
    sigmax = np.append(np.zeros([1, len(x) - len(sigmax)]), sigmax)

    # reverse the query and append zeros
    y = np.append(np.flip(query), np.zeros(pieces - m))

    step_size = k - m + 1
    stop = n - k + 1

    for j in range(0, stop, step_size):
        # The main trick of getting dot products in O(n log n) time
        X = np.fft.fft(x[j : j + k])
        Y = np.fft.fft(y)

        Z = X * Y
        z = np.fft.ifft(Z)

        d = 2 * (
            m
            - (z[m - 1 : k] - m * meanx[m + j - 1 : j + k] * meany)
            / (sigmax[m + j - 1 : j + k] * sigmay)
        )
        d = np.sqrt(d)
        dist = np.append(dist, d)

    j = j + k - m
    k = n - j - 1
    if k >= m:
        X = np.fft.fft(x[j : n - 1])
        y = y[0:k]

        Y = np.fft.fft(y)
        Z = X * Y
        z = np.fft.ifft(Z)

        d = 2 * (
            m
            - (z[m - 1 : k] - m * meanx[j + m - 1 : n - 1] * meany)
            / (sigmax[j + m - 1 : n - 1] * sigmay)
        )

        d = np.sqrt(d)
        dist = np.append(dist, d)

    return np.array(dist)
