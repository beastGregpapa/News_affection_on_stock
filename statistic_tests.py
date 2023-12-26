import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu, kruskal, ks_2samp

tickers = [
    "AAPL",
    "TSLA",
    "VEGI",
    "ILMN",
    "PFE",
    "DE"
]
cols = [
    "Ticker",
    "Params",
    "Params2",
    "Model",
    "test_rmse",
    "test_mae",
    "train_rmse",
    "train_mae",
    "test_predict",
    'PValueAvg',
    "Wilcoxon_PValue",
    "Kruskal_PValue",
    "MannWhitney_PValue",
    "Kolmogorov_PValue",
]
dfs = {}
for ticker in tickers:
    dfs[ticker] = pd.read_csv(f"tables/{ticker}.csv")

dfs_stats = []
for ticker, df_ticker in dfs.items():
    dfs_stats.append(pd.DataFrame(columns=cols))
    df_ticker.drop(columns=["Price", "TweetsToneAvg", "NewsToneAvg"], inplace=True)
    df_grouped = df_ticker.groupby(by=["Model"])
    for model, df_model in df_grouped:
        # df_model.test_predict = list(map(float, [x.strip('[]').split(', ') for x in df_model.test_predict]))
        df_model.test_predict = [list(map(float, x.strip('[]').split(', '))) for x in df_model.test_predict]
        for i1, line1 in df_model.iterrows():
            for i2, line2 in df_model.iterrows():
                if i1 == i2 or line1.test_predict == line2.test_predict:
                    continue
                print(f'{model[0]}  {line1.Params}/{line2.Params}')
                wilcoxon_stat = wilcoxon(line1.test_predict, line2.test_predict)
                kruskal_stat = kruskal(line1.test_predict, line2.test_predict)
                mannwhitneyu_stat = mannwhitneyu(line1.test_predict, line2.test_predict)
                ks_2samp_stat = ks_2samp(line1.test_predict, line2.test_predict)
                line1[f'PValueAvg'] = round((sum([wilcoxon_stat.pvalue, kruskal_stat.pvalue, mannwhitneyu_stat.pvalue,
                                           ks_2samp_stat.pvalue]) / 4), 4)
                line1[f'Wilcoxon_PValue'] = round(wilcoxon_stat.pvalue, 4)
                line1[f'Kruskal_PValue'] = round(kruskal_stat.pvalue, 4)
                line1[f'MannWhitney_PValue'] = round(mannwhitneyu_stat.pvalue, 4)
                line1[f'Kolmogorov_PValue'] = round(ks_2samp_stat.pvalue, 4)
                line1['Params2'] = line2.Params
                dfs_stats[-1].loc[len(dfs_stats[-1])] = line1
    dfs_stats[-1].drop(columns=["test_predict"], inplace=True)
    dfs_stats[-1].rename(columns={"Params": "Params1"}, inplace=True)
    dfs_stats[-1].to_csv(f"statistics/{ticker}.csv", float_format='%.40f')
