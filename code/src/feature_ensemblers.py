import pandas as pd
import numpy as np
import re
import logging
from src.feature_base import FeatureEnsemblerBase
# from ..util.decorators import log_status

__all__ = ['FeatureEnsembler']


class FeatureEnsembler(FeatureEnsemblerBase):
    def __init__(self,
                 featurizers,
                 fillna: str = 'linear',
                 label_col: str = 'load',
                 keep_filled_label: bool = False):
        """
        Ensemble features for bus load tasks.

        Args:
            featurizers: List of featurizer instances.
            fillna: df.interpolate(method=fillna)
            label_col: label column
            keep_filled_label: if False, will keep the original label column, otherwise, will
                               perform fillna for label column.
        """
        super().__init__(featurizers)
        self.fillna = fillna
        self.label_col = label_col
        self.keep_filled_label = keep_filled_label

#     @log_status()
    def transform(self, df: pd.DataFrame):
        """
        Transform df to df with various features defined by list of featurizers.

        Caution:
        1. Please ensure that input df has datetime index.

        Args:
            df: input df

        Returns: df with features ready for modeling

        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logging.warning('df index is not type of pd.DatetimeIndex, '
                            'this may lead to unexpected behavior in featurizers!')

        df = df.copy()
        label = df[self.label_col].copy()
        fill_cols = df.select_dtypes(include=[np.number]).columns
        df[fill_cols] = df[fill_cols].interpolate(method=self.fillna)

        logging.info('filling na for columns {} ...'.format(fill_cols))

        for featurizer in self.featurizers:
            df = featurizer.transform(df)

        df[fill_cols] = df[fill_cols].interpolate(method=self.fillna)

        # recover original label
        if not self.keep_filled_label:
            df[self.label_col] = label

        # replace symbols in column names
        df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x), inplace=True)

        return df
