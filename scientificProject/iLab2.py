import pandas as pd
from math import sqrt
from iLab_optimizer import *
from scipy.optimize import Bounds


def get_annual_mean_var_std(df):
    mean = df.mean() * 12
    variance = df.var() * 12
    std = df.std() * sqrt(12)
    return mean, variance, std


data_path = 'data/iLab2 assessment Data pool 2023 Term1.csv'
raw_data = pd.read_csv(data_path, index_col=0, parse_dates=True).apply(lambda x: x / 100)
new_col = []
for s in raw_data.columns:
    result = s.split('(')[1].split(')')[0]
    new_col.append(result)
raw_data.columns = new_col
SP50_return = raw_data.loc[:, 'SP50']
raw_data = raw_data.drop('SP50', axis=1, )
rf = 0.03
target_list = ['C-US', 'GOOGL-US', 'INTC-US', 'NVDA-US', 'EBAY-US', 'HD-US', 'IBM-US', 'JPM-US', 'WMT-US', 'CVX-US']
washed_df = raw_data.loc[:, target_list]
mean_list_annual, var_annual, annual_std = get_annual_mean_var_std(df=washed_df)
covariance_matrix = washed_df.cov()

guess = pd.Series([0.1 for i in range(10)], index=mean_list_annual.index)
portfolio_optimization_by_expected_return(target_function=Portfolio_std, mean_list=mean_list_annual,
                                          cov_matrix=covariance_matrix, start_guess=guess)
print('\nfinding p* which si to maximize sharpe ratio\n')

optimize_sharpe_ratio(mean_list_s=mean_list_annual,
                      cov_matrix_s=covariance_matrix,
                      start_guess=guess,
                      rf=rf,
                      bound=Bounds(0, 2))
