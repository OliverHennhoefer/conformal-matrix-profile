import os
import pandas as pd

from itertools import count
from datetime import datetime

from data_streaming.datasets import BenchmarkDataset
from data_streaming.streamer import CSVStreamer
from multiple_testing.benjamini_hochberg import BatchBH
from multiple_testing.benjamini_yekutieli import BatchBY

from conformal_matrix_profile.conformal_matrix_profile import (
    OnlineConformalMatrixProfile,
)

if __name__ == "__main__":
    dataset = CSVStreamer(dataset=BenchmarkDataset.MACKEY)
    #batch_by = BatchBY(alpha=0.2)
    batch_bh = BatchBH(alpha=0.2)

    damp = OnlineConformalMatrixProfile(
        subseq_len=60,
        window_len=5_000,
        calib_size=2_000,
        tail_frac=0.02,  # tail_frac: [30, 50]
        pieces=2**12,
    )

    warm_up_period = damp.window_len + damp.calib_size
    batch_size = 25
    results = [False] * warm_up_period
    estimates, min_dists, labels, batch = [], [], [], []

    for i, (value, label) in enumerate(dataset):
        pred, min_dist = damp.estimate_one(value)
        estimates.append(pred)
        min_dists.append(min_dist)
        labels.append(label)
        damp.learn_one(value)

        if i >= warm_up_period:
            batch.append(pred)

            if len(batch) == batch_size:
                #decisions = batch_by.test_batch(batch)
                decisions = batch_bh.test_batch(batch)
                results.extend(decisions)
                batch = []

    df = pd.DataFrame(
        {
            "prediction": estimates,
            "matrix_profile": min_dists,
            "decision": results,
            "label": labels,
        }
    )

    df["prediction"] = df["prediction"].apply(lambda x: x.real)

    for i in count():
        file_name = f"../results/exp{i}_{datetime.now().strftime("%y%m%d")}.csv"
        if not os.path.exists(file_name):
            break

    df.to_csv(file_name, index=False)  # noqa
    print(f"File saved as: {file_name}")
    print(pd.Series(results).value_counts())  # 39729 | 271
