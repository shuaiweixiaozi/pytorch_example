import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.stattools import acf

warnings.filterwarnings('ignore')


def sin_transform(values):
    return np.sin(2*np.pi*values/len(set(values)))


def cos_transform(values):
    return np.cos(2*np.pi*values/len(set(values)))


def get_yearly_autocorr(data):
    ac = acf(data, nlags=366)
    return (0.5 * ac[365]) + (0.25 * ac[364]) + (0.25 * ac[366])


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
test['sales'] = np.nan
data = pd.concat([train, test], ignore_index=True)
data = data[data['date'] > '2016-01-01']
data['store_item_id'] = data['store'].astype(str) + '_' + data['item'].astype(str)
data['dayofweek'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['day'] = data['date'].dt.day
data['year_mod'] = (data['year'] - data['year'].min()) / (data['year'].max() - data['year'].min())

data['dayofweek_sin'] = sin_transform(data['dayofweek'])
data['dayofweek_cos'] = cos_transform(data['dayofweek'])
data['month_sin'] = sin_transform(data['month'])
data['month_cos'] = cos_transform(data['month'])
data['day_sin'] = sin_transform(data['day'])
data['day_cos'] = cos_transform(data['day'])

data.drop('id', axis=1, inplace=True)
data = data.sort_values(['store_item_id', 'date'])
data = data[data['store_item_id'].isin(['1_1', '1_2'])]

train['store_item_id'] = train['store'].astype(str) + '_' + train['item'].astype(str)
mode = 'test'
if mode == 'valid':
    scale_data = train[train['date'] < '2017-01-01']
else:
    scale_data = train[train['date'] >= '2014-01-01']

scale_map = {}
scaled_data = pd.DataFrame()

for store_item_id, item_data in data.groupby('store_item_id', as_index=False):
    sidata = scale_data.loc[scale_data['store_item_id'] == store_item_id, 'sales']
    mu = sidata.mean()
    sigma = sidata.std()
    yearly_autocorr = get_yearly_autocorr(sidata)
    item_data.loc[:,'sales'] = (item_data['sales'] - mu) / sigma
    scale_map[store_item_id] = {'mu': mu, 'sigma': sigma}
    item_data['mean_sales'] = mu
    item_data['yearly_corr'] = yearly_autocorr
    scaled_data = pd.concat([scaled_data, item_data], ignore_index=True)

scaled_data['yearly_corr'] = ((scaled_data['yearly_corr'] - scaled_data['yearly_corr'].mean()) / scaled_data['yearly_corr'].std())
scaled_data['mean_sales'] = (scaled_data['mean_sales'] - scaled_data['mean_sales'].mean()) / scaled_data['mean_sales'].std()


scaled_data = reduce_mem_usage(scaled_data)
scaled_data.to_pickle('../data/processed_data_test_stdscaler.pkl')

