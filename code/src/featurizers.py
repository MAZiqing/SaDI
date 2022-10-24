import numpy as np
import pandas as pd
import logging
from typing import List, Dict
from chinese_calendar.constants import holidays, workdays
from chinese_calendar.utils import is_workday, is_holiday
from src.feature_base import FeaturizerBase, WinOffsetFeaturizerBase
# from ..util.decorators import log_status

__all__ = ['DateTimeFeaturizer', 'DifferenceFeaturizer', 'RollingStatsFeaturizer']


class DateTimeFeaturizer(FeaturizerBase):
    DEFAULT_FEATURES = [
        'year',
        'index_15min',
        'day_of_year',
        'is_leap_year',
        'day_of_year_sin',
        'day_of_year_cos',
        'quarter',
        'month',
        'week',
        'day',
        'day_of_month_sin',
        'day_of_month_cos',
        'day_of_week',
        'is_weekend',
        'day_of_week_sin',
        'day_of_week_cos',
        'hour',
        'minute',
        'second',
        'second_of_day',
        'second_of_day_sin',
        'second_of_day_cos',
        'is_workday',
        'is_holiday',
        'is_special_workday',
        'is_special_holiday'
    ]

    dt_format = '%Y-%m-%d %H:%M:%S'

    def __init__(self, *, feature_col: List[str] = None, dt_col: str = None):
        """
        featurizer for making Datetime features, see class attribute DEFAULT_FEATURES for more information.

        Args:
            feature_col: the list of datetime features interested, must in DEFAULT_FEATURES,
                         default using all DEFAULT_FEATURES
            dt_col: datetime column, default using the index of the input dataframe
        """
        super().__init__()

        if feature_col is None:
            self.feature_cols = self.DEFAULT_FEATURES
        else:
            self.feature_cols = list(feature_col)

        self.dt_col = dt_col

#     @log_status()
    def transform(self, df: pd.DataFrame):
        """
        Transform input df to df with specified datetime features.

        Caution:
        1. The input df should have index that is a pandas datetime index, if not, should specify dt_col.
        2. The time gap between two samples should be 15mins.
        3. The output df will have a datetime index using original df.index or dt_col

        Args:
            df: input df

        Returns: df with Datetime features

        """
        if self.dt_col is None:
            dt_col = df.index
        else:
            dt_col = df[self.dt_col]

        dt_col = pd.to_datetime(dt_col, format=self.dt_format)
        dt_col = dt_col.to_series()
        out_df = pd.DataFrame(index=dt_col)

        year = dt_col.dt.year
        doy = dt_col.dt.dayofyear
        leap = np.asarray(dt_col.dt.is_leap_year, dtype=int)
        diy = 365 + leap
        day = dt_col.dt.day
        dim = dt_col.dt.days_in_month
        dow = dt_col.dt.dayofweek
        doy_angular = 2 * np.pi * doy / diy
        day_angular = 2 * np.pi * day / dim
        dow_angular = 2 * np.pi * dow / 7

        out_df['year'] = year - year.min()
        out_df['day_of_year'] = doy
        out_df['is_leap_year'] = leap
        out_df['day_of_year_sin'] = np.sin(doy_angular)
        out_df['day_of_year_cos'] = np.cos(doy_angular)

        out_df['quarter'] = dt_col.dt.quarter
        out_df['month'] = dt_col.dt.month
        out_df['week'] = dt_col.dt.isocalendar().week

        out_df['day'] = day
        out_df['day_of_month_sin'] = np.sin(day_angular)
        out_df['day_of_month_cos'] = np.cos(day_angular)

        out_df['day_of_week'] = dow
        out_df['is_weekend'] = np.asarray(dow >= 5, dtype=int)
        out_df['day_of_week_sin'] = np.sin(dow_angular)
        out_df['day_of_week_cos'] = np.cos(dow_angular)

        out_df['hour'] = dt_col.dt.hour
        out_df['minute'] = dt_col.dt.minute
        out_df['index_15min'] = out_df['hour'] * 4 + out_df['minute'] // 15
        out_df['second'] = dt_col.dt.second
        out_df['second_of_day'] = out_df['hour'] * 3600 + out_df['minute'] * 60 + out_df['second']
        sod_angular = 2 * np.pi * out_df['second_of_day'] / 86400
        out_df['second_of_day_sin'] = np.sin(sod_angular)
        out_df['second_of_day_cos'] = np.cos(sod_angular)

        out_df['is_workday'] = dt_col.dt.date.map(is_workday)
        out_df['is_holiday'] = dt_col.dt.date.map(is_holiday)
        out_df['is_special_workday'] = dt_col.dt.date.map(lambda d: d in workdays)
        out_df['is_special_holiday'] = dt_col.dt.date.map(lambda d: d in holidays)
        out_df['is_special_workday'] = out_df['is_special_workday'].map(lambda x: 1 if x else 0)
        out_df['is_special_holiday'] = out_df['is_special_holiday'].map(lambda x: 1 if x else 0)

        return pd.concat([df, out_df[self.feature_cols]], axis=1)


class DifferenceFeaturizer(WinOffsetFeaturizerBase):

    def __init__(self, *, offsets, feature_col, freq=None):
        """
        Building difference features.

        Args:
            offsets: list of lags, each element in the list corresponds to one lag.
            feature_col: list of input raw features. These features will be used to calculate stats and lag features.
            freq: The shift frequency.
        """
        super().__init__(offsets=offsets, feature_col=feature_col)

        self.freq = freq

#     @log_status()
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform df to have difference features

        Args:
            df: input df

        Returns: output df with difference features.

        """
        feature_df = df[self.feature_col]
        df_lst = [df]

        for offset in self.offsets:
            diff_df = feature_df - feature_df.shift(offset, self.freq)
            diff_df.columns = [
                '{}_diff_offset_{}'.format(col, offset)
                for col in feature_df.columns.tolist()
            ]
            df_lst.append(diff_df)

        return pd.concat(df_lst, axis=1)


class RollingStatsFeaturizer(WinOffsetFeaturizerBase):
    DEFAULT_STATS = ['min', 'max', 'median', 'mean', 'std', 'skew']
    DEFAULT_ROLLING_KWARGS = {'min_periods': 1, 'center': False}

    def __init__(self,
                 *,
                 wins,
                 offsets,
                 feature_col,
                 is_interval: bool = False,
                 interval_key: str = 'index_15min',
                 freq: int = None,
                 stats: List[str] = None,
                 quantiles: List[float] = None,
                 rolling_kwargs: Dict = None):
        """
        Generate Rolling stats features, this is a class designed for three purpose:
        1. Generate Rolling window stats.
            1. subseries rolling window stats.
            2. ordinal rolling window stats.
        2. Generate Lag features.

        To generate pure Lag features, include window size 1 in wins, then this will generate pure lag according
        to offsets.

        To generate ordinal rolling window stats, set window size > 1, skew features will only be built when
        window size >= 3. The resulting ordinal rolling window stats will shift according to offsets. For example,
        for window size = 3 and the sample with timestamp 2020-01-02 08:30:00, the window will contain samples:

        ["2020-01-02 08:00:00", "2020-01-02 08:15:00", "2020-01-02 08:30:00"]


        To generate subseries rolling window stats, set is_interval=True, set window size > 1, skew features will
        only be built when window size >= 3. The data will first be grouped by using interval_key. The resulting ordinal
        rolling window stats will shift according to offsets. For example, for window size = 3 and the sample with
        timestamp 2020-01-05 08:30:00, the window will contain samples:

        ["2020-01-03 08:30:00", "2020-01-04 08:30:00", "2020-01-05 08:30:00"]

        Caution:
        if is_interval=True:
        1. Ensure that index is pd datetime index.
        2. Ensure that index gap is 15 mins.
        3. Ensure that index has no missing values or duplicate values.
        4. Ensure that the columns interested have no missing values.

        Args:
            wins: list of window sizes, each element in the list corresponds to one window size.
            offsets: list of lags, each element in the list corresponds to one lag.
            feature_col: list of input raw features. These features will be used to calculate stats and lag features.
            is_interval: If True, subseries rolling window stats else ordinal rolling stats features.
            interval_key: The interval key used to groupby df. used only if is_interval=True.
            freq: The shift frequency.
            stats: The stats features, if None, DEFAULT_STATS are used. skew is used only when window size >= 3, if
                   window size = 1, only mean will be used.
            quantiles: list of quantiles to build quantile features.
            rolling_kwargs: extra kwargs pass to df.rolling function.
        """
        super().__init__(wins=wins, offsets=offsets, feature_col=feature_col)

        self.freq = freq
        self.stats = stats
        self.offsets = offsets
        self.is_interval = is_interval
        self.interval_key = interval_key
        self.quantiles = quantiles
        self.rolling_kwargs = rolling_kwargs
        self.wins = wins

#     @log_status()
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform df to have rolling stats features

        Args:
            df: input df

        Returns: output df with rolling stats features.

        """
        drop_interval_key = False

        if self.is_interval:
            if self.interval_key not in list(df.columns):
                logging.info('interval key {} not in df columns, '
                             'using default index_15min as interval key...'.format(self.interval_key))

                self.interval_key = 'index_15min'
                index_15min = df.index.hour * 4 + df.index.minute // 15
                df = df.assign(index_15min=index_15min)
                drop_interval_key = True  # drop index_15min after transformation

            if self.interval_key not in self.feature_col:
                self.feature_col.append(self.interval_key)
            else:
                logging.warning('interval_key {} in feature_col, '
                                'this feature will not be used to'
                                ' calculate the interval features ...'.format(self.interval_key))

        feature_df = df[self.feature_col]
        df_lst = [df]

        for win in self.wins:
            if win <= 1:
                logging.warning('window size <= 1, this will build lags only ...')
                stats = ['mean']
                win = 1
            elif win < 3 and 'skew' in self.stats:
                logging.warning('window size {} < 3, '
                                'skew requires at least 3 samples, '
                                'it will not be calculated !'.format(win))
                stats = self.stats[:]
                stats.remove('skew')
            else:
                stats = self.stats

            for offset in self.offsets:
                if self.is_interval:
                    curr_df = self._build_interval_stats(feature_df, win, offset, stats)
                else:
                    curr_df = self._build_rolling_stats(feature_df, win, offset, stats)

                df_lst.append(curr_df)

        out_df = pd.concat(df_lst, axis=1)
        if drop_interval_key:
            out_df.drop(columns=self.interval_key, inplace=True)

        return out_df

    def _build_interval_stats(self, df, win, offset, stats):
        df_lst = []
        df_groupby = df.groupby(self.interval_key)
        for col in df.columns.tolist():
            if col != self.interval_key:
                agg_dict = {
                    f'{self.interval_key}_{col}_win_{win}_offset_{offset}_{stat}': stat
                    for stat in stats
                }
                stats_df = df_groupby[col].apply(lambda x: x.rolling(win, **self.rolling_kwargs).agg(agg_dict))
                df_lst.append(stats_df)

                if self.quantiles is not None:
                    for qtl in self.quantiles:
                        pct_df = df_groupby[col].apply(lambda x: x.rolling(win, **self.rolling_kwargs).quantile(qtl))
                        pct_df.name = f'{self.interval_key}_{col}_win_{win}_offset_{offset}_q{qtl}'
                        df_lst.append(pct_df)

        return pd.concat(df_lst, axis=1).shift(offset, self.freq)

    def _build_rolling_stats(self, df, win, offset, stats):
        df_lst = []
        for col in df.columns.tolist():
            if col != self.interval_key:
                agg_dict = {
                    f'{col}_win_{win}_offset_{offset}_{stat}': stat
                    for stat in stats
                }
                stats_df = df[col].rolling(win, **self.rolling_kwargs).agg(agg_dict)
                df_lst.append(stats_df)

                if self.quantiles is not None:
                    for qtl in self.quantiles:
                        pct_df = df[col].rolling(win, **self.rolling_kwargs).quantile(qtl)
                        pct_df.name = f'{col}_win_{win}_offset_{offset}_q{qtl}'
                        df_lst.append(pct_df)

        return pd.concat(df_lst, axis=1).shift(offset, self.freq)

    @property
    def stats(self):
        return self._stats

    @stats.setter
    def stats(self, value):
        if value is None:
            logging.warning('using all default stats {}'.format(self.DEFAULT_STATS))
            self._stats = self.DEFAULT_STATS
        else:
            self._stats = list(value)

    @property
    def rolling_kwargs(self):
        return self._rolling_kwargs

    @rolling_kwargs.setter
    def rolling_kwargs(self, value):
        if value is None:
            self._rolling_kwargs = self.DEFAULT_ROLLING_KWARGS
        elif not isinstance(value, dict):
            raise ValueError('rolling_kwargs should be a dict!')
        else:
            self._rolling_kwargs = value

    @property
    def quantiles(self):
        return self._quantiles

    @quantiles.setter
    def quantiles(self, value):
        if value is not None:
            if isinstance(value, (float, int)):
                self._quantiles = [value]
            else:
                self._quantiles = list(value)

            for qtl in self._quantiles:
                if qtl > 1 or qtl < 0:
                    raise ValueError('Percentiles have to be float between 0 and 1!')

        else:
            self._quantiles = value

