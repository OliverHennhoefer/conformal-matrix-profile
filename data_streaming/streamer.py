import abc
import csv

from data_streaming.datasets import BenchmarkDataset


class BaseStreamer(abc.ABC):
    def __iter__(self):
        raise NotImplementedError


class CSVStreamer(BaseStreamer):
    def __init__(self, dataset: BenchmarkDataset):
        self.file_path: str = dataset.value

    def __iter__(self):
        with open(self.file_path, "r", newline="") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                yield row


class TXTStreamer(BaseStreamer):
    def __init__(self, dataset: BenchmarkDataset):
        self.file_path: str = dataset.value

    def __iter__(self):
        with open(self.file_path, "r") as file:
            for line in file:
                yield float(line.strip())
