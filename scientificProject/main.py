from math import sqrt

import pandas as pd

from iLab_optimizer import portfolio_optimization_by_expected_return, Portfolio_std,optimize_return_fixed_srd_c

path_perating_data_file = 'data/testy_data.csv'
data_df = pd.read_csv(path_perating_data_file,
                      index_col=0,
                      parse_dates=True).astype(float)

data_df.drop(columns='SP50', inplace=True)

data_df = data_df.loc[:, ['C', 'GOOGL', 'INTC', 'NVDA', 'EBAY']] / 100

Sample_cov_matrix = data_df.cov()
variance_list_annual = data_df.var() * 12
mean_list_annual = data_df.mean() * 12
std_list_annual = data_df.std() * sqrt(12)
# std_list_annual = std_list_annual.loc[['C', 'GOOGL', 'INTC', 'NVDA', 'EBAY']]

# Minimum Variance Frontier (MVF)
# minimum attainable standard deviation of annual returns for

W = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4], index=['C', 'GOOGL', 'INTC', 'NVDA', 'EBAY'])
target_return = 0.100
constrain_expected_reutrn = {'type': 'eq',  # constrain target return
                             'fun': lambda x: sum(mean_list_annual * x) - target_return}
opt_constraints = ({'type': 'eq',
                    'fun': lambda x: sum(x) - 1}
    # w adds up to 1, don't delete

    )

# portfolio_optimization_by_expected_return(target_function=Portfolio_std,
#                                           mean_list=mean_list_annual,
#                                           cov_matrix=Sample_cov_matrix,
#                                           constraints=opt_constraints,
#                                           start_guess=W)

# %% Investor J’s
# Optimal Complete Portfolio
# has an annualised standard deviation of 10%. What is Investor J’s
# 40. optimal allocation to risky assets 𝑦∗?
# 41. risk aversion coefficient, A?
# 42. Optimal Complete Portfolio annualised expected return?
# 43. Optimal Complete Portfolio annualised Sharpe ratio?
# 44. What is Investor J’s Optimal Complete Portfolio utility score?
# %% optimise Er-rf
x0_args = [0.2, 0.2, 0.2, 0.2, 0.2, 13.000]

# portfolio_return
#     Y = portfolio return- rf over A times portfolio variance
#     complete_std =
rf = 0.03


# def negative_complete_return(args, mean_list=mean_list_annual, cov_matrix=Sample_cov_matrix, rf=0.03):
#     w = args[0:5]
#     a = args[-1] * 50
#     portfolio_return = sum(mean_list * w)
#     R = portfolio_return - rf
#     portfolio_variance = pow(Portfolio_std(w, cov_matrix), 2)
#     denominator = a * portfolio_variance
#     y = R / denominator
#
#     weighted_risky_portfolio_return = y * portfolio_return
#
#     weighted_rf_return = (1 - y) * rf
#
#     complete_return = weighted_rf_return + weighted_risky_portfolio_return
#     return - complete_return


def complete_std(args, var_matrix):
    w = args[0:5]
    a = args[-1] * 50
    portfolio_return = sum(mean_list_annual * w)
    R = portfolio_return - rf
    portfolio_variance = pow(Portfolio_std(w, var_matrix), 2)
    y = R / (a * portfolio_variance)

    return y * sqrt(portfolio_variance)


opt_constraints = ({'type': 'eq',
                    'fun': lambda args: sum(args[0:5]) - 1}
                   # w adds up to 1, don't delete
                   , {'type': 'eq',
                      'fun': lambda args: complete_std(args, Sample_cov_matrix) - 0.10 + 0.00000000001
                      # complete standard deviation is 0.1
                      }
                   )
# res = scipy.optimize.minimize(negative_complete_return,
#                               args=(mean_list_annual, Sample_cov_matrix),
#                               constraints=opt_constraints,
#                               x0=x0_args,
#                               bounds=scipy.optimize.Bounds(-2, 2))
# print(res.x, sum(res.x[0:5]))
# portfolio_w_res = pd.Series(res.x[0:5]
#                             , index=mean_list_annual.index
#                             )
#
# complete_array = res.x

# output_w = [-60.7908298556088,
#             38.140935173185,
#             68.895682399312,
#             101.622690160194,
#             -47.868477877082]
# output_w = [x/100 for x in output_w]


# output_w = portfolio_w_res
#
# test_complete_array = complete_array
#
# portfolio_return = sum(mean_list_annual * output_w)
# portfolio_std = Portfolio_std(output_w, Sample_cov_matrix)
# sharpe_portfolio = (portfolio_return - rf) / portfolio_std
# print('portfolio std is ', portfolio_std)
# print('portfolio return is ', portfolio_return)
# print('portfolio sharpe is ', sharpe_portfolio)
#
# c_std = complete_std(test_complete_array, Sample_cov_matrix)
# c_return = -negative_complete_return(test_complete_array, mean_list_annual, Sample_cov_matrix)
# sharpe_complete = (c_return - rf) / c_std
# print('complete return is ', c_return)
# print('complete std is ', c_std)
# print('complete sharpe is ', sharpe_complete)
#
# Y = c_std / portfolio_std
# print('y is', Y)
#%%
