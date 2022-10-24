from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from typing import List, Union
import pandas as pd


class FeaturizerBase(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def copy(self):
        return deepcopy(self)


class FeatureEnsemblerBase(metaclass=ABCMeta):

    def __init__(self, featurizers: List[FeaturizerBase]):
        """
        Ensemble featurizers. Featurizers should be a list of instance of FeaturizerBase

        Args:
            featurizers: List of instance of FeaturizerBase
        """
        self.featurizers = featurizers

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    def featurizers(self):
        return self._featurizers

    @featurizers.setter
    def featurizers(self, value):
        if not isinstance(value, Sequence):
            value = [value]
        else:
            value = list(value)

        for featurizer in value:
            if not isinstance(featurizer, FeaturizerBase):
                raise TypeError('{} should be a instance of FeaturizerBase'.format(featurizer))

        self._featurizers = value

    def copy(self):
        return deepcopy(self)


class WinOffsetFeaturizerBase(FeaturizerBase, metaclass=ABCMeta):

    def __init__(self,
                 wins: Union[int, List[int]] = None,
                 offsets: Union[int, List[int]] = None,
                 feature_col: List[str] = None):
        """
        Window offset featurizer base, the child class should have at least one of the following attributes:
        1. wins
        2. offsets

        Args:
            wins: list of window sizes
            offsets: list of offsets or lags
            feature_col: list of feature columns
        """
        super().__init__()

        self.wins = wins
        self.offsets = offsets
        self.feature_col = feature_col

    @property
    def wins(self):
        return self._wins

    @wins.setter
    def wins(self, value):
        if value is None:
            self._wins = []
        elif isinstance(value, int):
            self._wins = [value]
        else:
            self._wins = list(value)

    @property
    def offsets(self):
        return self._offsets

    @offsets.setter
    def offsets(self, value):
        if value is None:
            self._offsets = []
        elif isinstance(value, int):
            self._offsets = [value]
        else:
            self._offsets = list(value)

    @property
    def feature_col(self):
        return self._feature_col

    @feature_col.setter
    def feature_col(self, value):
        if not value:
            raise ValueError('feature_col can not be empty!')
        elif isinstance(value, str):
            self._feature_col = [value]
        else:
            self._feature_col = list(value)

