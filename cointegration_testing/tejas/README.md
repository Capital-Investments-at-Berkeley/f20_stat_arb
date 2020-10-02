## Tejas - Phillips and Ouliaris Test

Philips Oularis Test: Residual-based test for cointegration -> applies unit root tests to residuals from SOLS estimation. 

Essentially same as Engle-Granger, but uses Phillips Perron instead of ADF for unit root testing. Phillips Perron test is more robust to autocorrelation in the dataset but performs worse in finite samples (Davidson and MacKinnon, 2004).

Would be interesting to compare the results of Phillips and Oularis Test with Engle-Granger to assess impact of noise due to autocorrelation on individual time series. 