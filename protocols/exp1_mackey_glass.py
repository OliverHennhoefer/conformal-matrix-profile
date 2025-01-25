import os
import pandas as pd

from datetime import datetime

from data_streaming.datasets import BenchmarkDataset
from data_streaming.streamer import CSVStreamer
from multiple_testing.benjamini_yekutieli import BatchBY

from conformal_matrix_profile.conformal_matrix_profile import (
    OnlineConformalMatrixProfile,
)

if __name__ == "__main__":
    dataset = CSVStreamer(dataset=BenchmarkDataset.MACKEY)
    batch_by = BatchBY(alpha=0.2)

    damp = OnlineConformalMatrixProfile(
        subseq_len=60, window_len=5_000, calib_size=2_000, tail_frac=0.02, pieces=2**12  # tail_frac: [30, 50]
    )

    warm_up_period = 9_999
    batch_size = 25
    results = [False] * warm_up_period

    estimates = []
    labels = []
    batch = []

    for i, instance in enumerate(dataset):
        value = float(instance["value"])
        pred = damp.estimate_one(value)
        estimates.append(pred)
        labels.append(int(instance["is_anomaly"]))
        damp.learn_one(value)

    for i in range(warm_up_period, len(estimates), batch_size):
        batch = estimates[i : i + batch_size]
        result = batch_by.test_batch(batch)
        results.extend(result)

    df = pd.DataFrame(
        {
            "prediction": estimates,
            "decision": results,
            "label": labels,
        }
    )

    df["prediction"] = df["prediction"].apply(lambda x: x.real)

    date_abbrev = datetime.now().strftime("%y%m%d")
    base_name = "exp"
    counter = 1

    while True:
        file_name = f"../results/{base_name}{counter}_{date_abbrev}.csv"
        if not os.path.exists(file_name):
            break
        counter += 1

    df.to_csv(file_name, index=False)
    print(f"File saved as: {file_name}")
    print(pd.Series(results).value_counts())  # 39812 | 188
