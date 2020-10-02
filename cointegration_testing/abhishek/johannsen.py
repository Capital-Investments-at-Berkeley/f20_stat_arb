from pandas import DataFrame
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from yfinance import download


class Johannsen():
    def __init__(self):
        pass

    def run_test(self, df: DataFrame):
        '''
        Completes a Johannsen Test of Cointegration on DataFrame
        Params:
            df: pd.DataFrame
        Return: statsmodels.tsa.vector_ar.vecm.JohansenTestResult
        '''
        assert len(df.columns) > 1
        jres = coint_johansen(df, det_order=0, k_ar_diff=1)
        return jres

    def asset_test(
        self, tickers: list,
        start_date: str,
        end_date: str,
        frequency: str,
        value: str = 'Close'
    ):
        '''
            Completes a Johannsen Test on N Assets
            Params:
                tickers: List of strings of tickers
                start_date: beginning day of data
                end_date: ending day of data
                frequency: how often to take values in the interval
                value: type of value to pull (ie Close)
            Return: statsmodels.tsa.vector_ar.vecm.JohansenTestResult
        '''
        asset_df = DataFrame()
        for ticker in tickers:
            asset_df[ticker] = download(
                ticker,
                start=start_date,
                end=end_date,
                interval='1d'
            )[value]
        return self.run_test(asset_df)

    def print_result(self, jres):
        '''
            Prints Significant values of a Johannsen Test
            Params:
                jres: statsmodels.tsa.vector_ar.vecm.JohansenTestResult
            Return: None
        '''
        print("Trace Stat: ", jres.trace_stat)
        print("Critical Trace Stat: ", jres.trace_stat_crit_vals)
        print("Max Eigenvalue Stat: ", jres.max_eig_stat)
        print("Critical Max Eigenvalue Stat: ", jres.max_eig_stat_crit_vals)


def johannsen_example():
    tickers = ['KO', 'PEP']
    johannsen = Johannsen()
    results = johannsen.asset_test(tickers, '2015-09-01', '2020-09-29')
    johannsen.print_result(results)


if __name__ == '__main__':
    johannsen_example()
