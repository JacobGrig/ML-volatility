import numpy as np
import pandas as pd
import abc

from tqdm import tqdm
from pathlib import Path


class BaseAlgorithm(metaclass=abc.ABCMeta):

    def __init__(self):
        self.recognized_text_dict = {}
        self.true_text_dict = {}

    @abc.abstractmethod
    def b(self):
        pass


class GarchAlgorithm(BaseAlgorithm):
    def __init__(self):
        super().__init__()
        self.ocr_df = pd.DataFrame({})
        self.pick_df = pd.DataFrame({})
        self.entity_dict = {}


if __name__ == "__main__":
    pass
