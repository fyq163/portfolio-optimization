from numpy import array
from math import sqrt
import pandas as pd
import scipy
from scipy.optimize import Bounds


def get_annual_mean_var_std(df):
    mean = df.mean() * 12
    variance = df.var() * 12
    std = df.std() * sqrt(12)
    return mean, variance, std


def Portfolio_std(w, cov_matrix):
    """

    :param w: takes pandas series
    :param cov_matrix: same order covariance matrx
    :return: float, portfolio standard deviation
    """
    cov_np_array = array(cov_matrix)
    w_np = array(w)
    # if len(w)!=len(cov_matrix.index):
    #     raise 'not right'
    length = len(cov_matrix.index)
    p_variance = 0
    for i in range(length):
        for j in range(length):
            p_variance += w_np[i] * w_np[j] * cov_np_array[i, j]
    return sqrt(p_variance * 12)


def portfolio_optimization_by_expected_return(target_function,
                                              mean_list,
                                              cov_matrix,
                                              start_guess,
                                              bound=Bounds(-2, 2),
                                              target_return=0.100,
                                              mode='GMVP'):
    """
    :param target_return:
    :param mode: 'GMVP' or 'fix_return' must give target return
    :param target_function: compulsory, what you want to optimise no (), like function=func, if sharpe ratio should be negative sharpe ratio
    :param mean_list: compulsory,annually expected return
    :param cov_matrix: compulsory
    :param start_guess: vector of x0, must be pd.Series
    :param bound:
    :return:This function is general
    """
    if any(mean_list.index != start_guess.index):
        # test if index equal
        print(False)
    if mode not in ['GMVP', 'fix_return']:
        raise ValueError('mode not allowed')
    else:
        if mode == 'GMVP':
            mode_func = {'type': 'ineq',
                         'fun': lambda x: sum(x)}
        elif mode == 'fix_return':
            if target_return:
                print('your did not give target return, using default rate 0.1 ')
            mode_func = {'type': 'ineq',
                         'fun': lambda x: sum(mean_list * x) - target_return}

    opt_constraints = ({'type': 'eq',
                        'fun': lambda x: sum(x) - 0.99999999}
                       # w adds up to 1, don't delete
                       , mode_func
                       )

    result = scipy.optimize.minimize(
        lambda w: target_function(w, cov_matrix=cov_matrix),
        x0=array(start_guess),
        constraints=opt_constraints,
        bounds=bound

    )

    w_res = pd.Series(result.x, index=start_guess.index).round(4)
    # return calculation
    weighted_return = sum(mean_list * w_res)

    print(result.message, '\nPortfolio Weighting is \n', w_res,
          '\nwhen weighted_return is ', round(weighted_return, 4), 'at standard deviation of ',
          round(result.fun, 4))


def optimize_sharpe_ratio(mean_list_s, cov_matrix_s, start_guess, rf: int, bound=Bounds(-2, 2)):
    if any(mean_list_s.index != start_guess.index):
        # test if index equal
        raise ValueError('mean and star guess not right')

    def negative_sharpe_ratio(w):
        sharpe_ratio = 0.00000000 - ((sum(mean_list_s * w) - rf) / Portfolio_std(w, cov_matrix_s))
        return sharpe_ratio

    result = scipy.optimize.minimize(negative_sharpe_ratio,
                                     x0=array(start_guess),
                                     bounds=bound
                                     )
    constraints = {'type': 'eq',
                   'fun': lambda x: sum(x) - 0.99999999}
    w_res = pd.Series(result.x, index=start_guess.index).round(4)
    weighted_return = sum(mean_list_s * w_res)

    print(result.message, '\nPortfolio Weighting is \n', w_res,
          '\nwhen weighted_return is ', round(weighted_return, 4), 'at standard deviation of ',
          Portfolio_std(w_res, cov_matrix_s),
          'with sharpe ratio =', -(negative_sharpe_ratio(w_res)))


def optimize_return_fixed_srd_c(mean_list, cov_matrix, target_complete_std, rf=0.03):
    """

    :param mean_list: a list of expected return with
    :param cov_matrix:
    :param rf: risk free rate
    :param target_complete_std: what target complete_std your want to give
    :return: Print out result and give a array of result
    """
    # Gnerate guess array
    x0 = []
    number_of_stocks = len(mean_list.index)
    equal_weight = 1 / number_of_stocks
    for i in range(number_of_stocks):
        x0 = x0 + [equal_weight]
    x0 = x0 + [0.01]

    # target-negative returns of complete portfolio

    def give_portfolio_return(args, mean):
        return sum(mean * args[0:-1])

    def give_portfolio_variance(args, variance_matrix):
        w = args[0:-1]
        return pow(Portfolio_std(w, variance_matrix), 2)

    def give_R(args, mean):
        portfolio_return = give_portfolio_return(args, mean)
        R = portfolio_return - rf
        return R

    def give_y(args, mean, cov):
        """

        :param args: input of the whole optimize porcess
        :param mean: mean return list of all stocks
        :param cov: covariance matrix of all stocks
        :return: optimal y which is percentage of risky asset in complete portfolio
        """
        a = args[-1] * 100
        R = give_R(args, mean)
        portfolio_variance = give_portfolio_variance(args, cov)
        denominator = a * portfolio_variance
        y = R / denominator
        return y

    def negative_complete_return(args, mean, cov, rf):
        y = give_y(args, mean, cov)
        portfolio_return = give_portfolio_return(args, mean)
        weighted_risky_portfolio_return = y * portfolio_return

        weighted_rf_return = (1 - y) * rf

        complete_return = weighted_rf_return + weighted_risky_portfolio_return
        return - complete_return

    def complete_std(args, mean, cov_matrix):
        y = give_y(args, mean=mean, cov=cov_matrix)
        portfolio_variance = give_portfolio_variance(args, cov_matrix)
        return y * sqrt(portfolio_variance)

    opt_constraints = ({'type': 'eq',
                        'fun': lambda args: sum(args[0:5]) - 1}
                       # w adds up to 1, don't delete
                       , {'type': 'eq',
                          'fun': lambda args:
                          complete_std(args, mean_list, cov_matrix) - target_complete_std + 0.00000000001
                          # complete standard deviation is 0.1
                          }
                       )

    res = scipy.optimize.minimize(negative_complete_return,
                                  args=(mean_list, cov_matrix, rf),
                                  constraints=opt_constraints,
                                  x0=x0,
                                  bounds=scipy.optimize.Bounds(-2, 2))
    answer = pd.Series(res.x[0:-1], index=mean_list.index).apply(lambda x: str(x * 100))
    answer.loc['a'] = res.x[-1] * 100
    print(negative_complete_return(res.x, mean_list_annual, Sample_cov_matrix, 0.03))
    return answer


if __name__ == '__main__':
    path_perating_data_file = 'data/testy_data.csv'
    data_df = pd.read_csv(path_perating_data_file,
                          index_col=0,
                          parse_dates=True).astype(float)

    data_df.drop(columns='SP50', inplace=True)

    data_df = data_df.loc[:, ['C', 'GOOGL', 'INTC', 'NVDA', 'EBAY']] / 100

    Sample_cov_matrix = data_df.cov()
    variance_list_annual = data_df.var() * 12
    mean_list_annual = data_df.mean() * 12
    mean_list = mean_list_annual
    std_list_annual = data_df.std() * sqrt(12)
    # std_list_annual = std_list_annual.loc[['C', 'GOOGL', 'INTC', 'NVDA', 'EBAY']]

    # Minimum Variance Frontier (MVF)
    # minimum attainable standard deviation of annual returns for

    W = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4], index=['C', 'GOOGL', 'INTC', 'NVDA', 'EBAY'])

    print('GMVP\n')
    portfolio_optimization_by_expected_return(target_function=Portfolio_std, mean_list=mean_list_annual,
                                              cov_matrix=Sample_cov_matrix, start_guess=W)
    # print('highest sharpe\n')
    # optimize_sharpe_ratio(mean_list_annual,
    #                       Sample_cov_matrix,
    #                       opt_constraints,
    #                       start_guess=W,
    #                       rf=0.03)
    # # %% Test optimize by fixed complete portfolio
    # re = optimize_return_fixed_srd_c(mean_list=mean_list_annual,
    #                                  cov_matrix=Sample_cov_matrix,
    #                                  target_complete_std=0.1
    #                                  )
    # print(re)
