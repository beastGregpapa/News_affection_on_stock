import pandas as pd
import json

with open("results.json", 'r', encoding='utf8') as f:
    json_data = f.read()

data = json.loads(json_data)
# df = pd.json_normalize(data, max_level=3)
# print(df.head(10))
cols = [
    "Ticker",
    "Params",
    "Model",
    "Price",
    "TweetsToneAvg",
    "NewsToneAvg",
    "test_rmse",
    "test_mae",
    "train_rmse",
    "train_mae",
    "test_predict"
]
print("START")
df = pd.DataFrame(columns=cols)
for ticker, data_ticker in data.items():
    for param, data_param in data_ticker.items():
        for model, data_model in data_param['models'].items():
            # data_model.model.save(f'{ticker}_{model}_{param}.h5')
            # df = pd.concat([df, pd.DataFrame([
            #     ticker,
            #     param,
            #     model,
            #     data_model['stats']['test_rmse'],
            #     data_model['stats']['test_mae'],
            #     data_model['stats']['train_rmse'],
            #     data_model['stats']['train_mae'],
            #     data_model['stats']['test_predict'],
            # ], columns=cols)], ignore_index=True)
            price_param = "Price" in param
            tweets_tone_avg_param = "TweetsToneAvg" in param
            news_tone_avg_param = "NewsToneAvg" in param
            new_row = [
                ticker,
                param,
                model,
                price_param,
                tweets_tone_avg_param,
                news_tone_avg_param,
                data_model['stats']['test_rmse'],
                data_model['stats']['test_mae'],
                data_model['stats']['train_rmse'],
                data_model['stats']['train_mae'],
                data_model['stats']['test_predict'],
            ]
            df.loc[len(df)] = new_row
df.to_csv("results.csv", index=False)
dfs = df.groupby(by=["Ticker"])
for ticker, ticker_df in dfs:
    # ticker = str
    ticker_df.to_csv(f"tables/{ticker}.csv", index=False)
print("END")

