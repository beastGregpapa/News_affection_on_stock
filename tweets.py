# -*- coding: utf-8 -*-

# -- Sheet --

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras import callbacks
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
from typing import List, Optional

RANDOM_STATE = 21072001


class TickerAnalyzer:
    def __init__(self, ticker="AAPL", news_path="news.csv", tweets_path="tweets.csv"):
        self.ticker = ticker
        self.news_df = pd.read_csv(news_path)
        self.news_df = self.news_df[self.news_df.Ticker == self.ticker]
        self.prices_df = pd.read_csv(f"prices/{ticker}.csv")
        self.tweets_df = pd.read_csv(tweets_path)
        self.news_df = self.news_df[self.news_df.Ticker == self.ticker]
        self.news_df['Date'] = pd.to_datetime(self.news_df['Date'], format='%Y-%m-%dT%H:%M:%S%z')
        # self.news_df['Date'] = self.news_df['Date'].dt.floor('D')
        self.news_df['Date'] = self.news_df['Date'].apply(lambda x: int(x.timestamp()) // 86400 * 86400)
        self.prices_df['Date'] = pd.to_datetime(self.prices_df['Date'], unit='s')
        self.prices_df['Date'] = self.prices_df['Date'].apply(lambda x: int(x.timestamp()) // 86400 * 86400)
        self.news_df['Tone'] = self.news_df['Tone'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
        self.tweets_df['Tone'] = self.tweets_df['Tone'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
        self.df = None

    def uniteData(self):
        dates = self.prices_df['Date'].unique()
        avg_sentiment_news = []
        pos_count_news = []
        neu_count_news = []
        neg_count_news = []
        avg_sentiment_tweets = []
        pos_count_tweets = []
        neu_count_tweets = []
        neg_count_tweets = []
        prices = []
        for date in dates:
            temp_prices = self.prices_df[self.prices_df['Date'] == date]
            temp_tweets = self.tweets_df[self.tweets_df['Date'] == date]
            _val = temp_tweets['Tone'].mean()
            avg_sentiment_tweets.append(_val if not np.isnan(_val) else 0)
            pos_count_tweets.append(len(temp_tweets[temp_tweets['Tone'] == 1]))
            neu_count_tweets.append(len(temp_tweets[temp_tweets['Tone'] == 0]))
            neg_count_tweets.append(len(temp_tweets[temp_tweets['Tone'] == -1]))

            prices.append(temp_prices['adjClose'].mean())

            temp_news = self.news_df[self.news_df['Date'] == date]
            _val = temp_news['Tone'].mean()
            avg_sentiment_news.append(_val if not np.isnan(_val) else 0)
            pos_count_news.append(len(temp_news[temp_news['Tone'] == 1]))
            neu_count_news.append(len(temp_news[temp_news['Tone'] == 0]))
            neg_count_news.append(len(temp_news[temp_news['Tone'] == -1]))
        self.df = pd.DataFrame({
            'Date': dates,
            'TweetsToneAvg': avg_sentiment_tweets,
            'TweetsTonePositive': pos_count_tweets,
            'TweetsToneNeutral': neu_count_tweets,
            'TweetsToneNegative': neg_count_tweets,
            'NewsToneAvg': avg_sentiment_news,
            'NewsTonePositive': pos_count_news,
            'NewsToneNeutral': neu_count_news,
            'NewsToneNegative': neg_count_news,
            'Price': prices,
        })
        self.df['Date'] = pd.to_datetime(self.df['Date'], unit="s")
        self.df = self.df.sort_values(by='Date')
        self.df.reset_index(drop=True, inplace=True)
        self.df['Price'] = self.df['Price'].astype('float32')
        self.df['Change'] = (self.df['Price'] - self.df['Price'].shift(1))
        self.df['Change'].fillna(0, inplace=True)
        print(f"[{self.ticker}] Df Shape: {self.df.shape}")


tickers = [
    "AAPL",
    "TSLA",
    "ILMN",
    "VEGI",
    "PFE",
    "DE",
]
analyzers = [TickerAnalyzer(ticker) for ticker in tickers]
for _ in range(len(analyzers)):
    plt.figure(figsize=(10, 10))
    plt.title(f"Correlation Heatmap {analyzers[_].ticker}")
    analyzers[_].uniteData()
    corr_matrix = analyzers[_].df.corr()
    # corr_matrix
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.savefig(f"{analyzers[_].ticker}_correlation.png")
    # plt.show()


class Model:
    def __init__(self, df, filters=[]):
        self.data = df[filters]
        self.filters = filters
        print("Init")
        self.X_train = None
        self.y_train = None
        self.X_Test = None
        self.y_test = None
        self.model = None
        self.history = None
        self.scaler = None
        self.test_mae = 0
        self.test_rmse = 0
        self.train_mae = 0
        self.train_rmse = 0
        self.train_predict = None
        self.test_predict = None

    def fit(self):
        pass

    def draw_train_graph(self, ticker=None, ax=None):
        # plt.figure(figsize=(6, 4))

        Xt = self.model.predict(self.X_train)
        Xt = Xt.flatten()

        df_actual_keys = ["Actual"]
        if len(self.filters) > 1:
            df_actual_keys += ['null_val']
        df_actual_keys += [f"null_val{i}" for i in range(1, len(self.filters) - 1)]
        df_actual_data = {"Actual": self.y_train}
        for _key in df_actual_keys[1:]:
            df_actual_data[_key] = [0] * len(self.y_train)

        df_predicted_keys = ["Predicted"]

        if len(self.filters) > 1:
            df_predicted_keys.append('null_val')

        df_predicted_keys += [f"null_val{i}" for i in range(1, len(self.filters) - 1)]
        df_predicted_data = {"Predicted": Xt}
        for _key in df_predicted_keys[1:]:
            df_predicted_data[_key] = [0] * len(Xt)

        df_actual = pd.DataFrame(df_actual_data)
        df_actual[df_actual_keys] = self.scaler.inverse_transform(df_actual[df_actual_keys])
        # print(self.scaler.inverse_transform(df_actual[df_actual_keys]))

        df_predicted = pd.DataFrame(df_predicted_data)
        # print(self.scaler.inverse_transform(df_predicted[df_predicted_keys]))
        df_predicted[df_predicted_keys] = self.scaler.inverse_transform(df_predicted[df_predicted_keys])
        self.train_predict = df_predicted.Predicted

        ax.plot(df_actual.Actual, label="Actual")
        ax.plot(self.train_predict, label="Predicted")
        ax.legend()

        ax.set_title(f"Train Dataset [{self.__class__.__name__}] [{self.filters}] [{ticker}]")
        self.train_rmse = math.sqrt(mean_squared_error(df_actual.Actual, self.train_predict))
        self.train_mae = mean_absolute_error(df_actual.Actual, self.train_predict)
        print(f"[{self.filters}] [{self.__class__.__name__}] Train RMSE =", self.train_rmse)
        print(f"[{self.filters}] [{self.__class__.__name__}] Train MAE =", self.train_mae)

    def draw_test_graph(self, ticker=None, ax=None):
        # plt.figure(figsize=(6, 4))
        Xt = self.model.predict(self.X_Test)
        Xt = Xt.flatten()

        df_actual_keys = ["Actual"]
        df_actual_data = {"Actual": self.y_test}
        if len(self.filters) > 1:
            df_actual_keys.append('null_val')
        df_actual_keys += [f"null_val{i}" for i in range(1, len(self.filters) - 1)]
        for _key in df_actual_keys[1:]:
            df_actual_data[_key] = [0] * len(self.y_test)

        df_predicted_keys = ["Predicted"]
        df_predicted_data = {"Predicted": Xt}

        if len(self.filters) > 1:
            df_predicted_keys.append('null_val')
        df_predicted_keys += [f"null_val{i}" for i in range(1, len(self.filters) - 1)]
        for _key in df_predicted_keys[1:]:
            df_predicted_data[_key] = [0] * len(Xt)

        df_actual = pd.DataFrame(df_actual_data)
        df_actual[df_actual_keys] = self.scaler.inverse_transform(df_actual[df_actual_keys])
        # print(self.scaler.inverse_transform(df_actual[df_actual_keys]))

        df_predicted = pd.DataFrame(df_predicted_data)
        # print(self.scaler.inverse_transform(df_predicted[df_predicted_keys]))
        df_predicted[df_predicted_keys] = self.scaler.inverse_transform(df_predicted[df_predicted_keys])
        self.test_predict = df_predicted.Predicted
        ax.plot(df_actual.Actual, label="Actual")
        ax.plot(self.test_predict, label="Predicted")
        ax.legend()

        ax.set_title(f"Test Dataset [{self.__class__.__name__}] [{self.filters}] [{ticker}]")
        self.test_rmse = math.sqrt(mean_squared_error(df_actual.Actual, self.test_predict))
        self.test_mae = mean_absolute_error(df_actual.Actual, self.test_predict)
        print(f"[{self.filters}] [{self.__class__.__name__}] Test RMSE =", self.test_rmse)
        print(f"[{self.filters}] [{self.__class__.__name__}] Test MAE =", self.test_mae)

    def plot_multiple_graphs(self, ticker=None):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

        # axs[0].set_title('Train Dataset')
        self.draw_train_graph(ticker, axs[0])

        # axs[1].set_title('Test Dataset')
        self.draw_test_graph(ticker, axs[1])

        plt.tight_layout()
        plt.savefig(f"graphs/{ticker}_{self.__class__.__name__}_{'_'.join(self.filters)}.png")
        # plt.show()

    def export(self, metric_params: Optional[List] = None):
        if not metric_params:
            result = {
                "test_rmse": float(self.test_rmse),
                "test_mae": float(self.test_mae),
                "train_rmse": float(self.train_rmse),
                "train_mae": float(self.train_mae),
                "test_predict": list(map(float, self.test_predict.tolist())),
                # "train_predict": self.train_predict.tolist(),
            }
            # print(result)
            return result
        result = {}
        for metric_param in metric_params:
            result[metric_param] = getattr(self, metric_param)
        return result


class LSTM_MODEL(Model):
    def __init__(self, df, filters):
        super().__init__(df, filters=filters)

    def process_data(self, lookback, train_size, scaler=StandardScaler):
        data = self.data.copy()

        train_data, test_data = train_test_split(data, test_size=1 - train_size, shuffle=False)

        # self.scaler = MinMaxScaler()
        # self.scaler = StandardScaler()
        self.scaler = scaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        X_train, y_train = [], []
        for i in range(len(train_data) - lookback - 1):
            X_train.append(train_data[i:i + lookback])
            y_train.append(train_data[i + lookback, 0])
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)

        # Process the testing data
        X_test, y_test = [], []
        for i in range(len(test_data) - lookback - 1):
            X_test.append(test_data[i:i + lookback])
            y_test.append(test_data[i + lookback, 0])
        self.X_Test, self.y_test = np.array(X_test), np.array(y_test)
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], len(self.filters)))
        print(f"self.X_Test shape: {self.X_Test.shape}")
        print(f"len(self.filters): {len(self.filters)}")
        self.X_Test = self.X_Test.reshape((self.X_Test.shape[0], self.X_Test.shape[1], len(self.filters)))

    def fit(self, lookback=10, train_size=0.8, scaler=StandardScaler, epochs=300):
        self.process_data(lookback, train_size, scaler)
        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        self.model = Sequential()

        self.model.add(LSTM(units=256, input_shape=(lookback, len(self.filters))))
        # self.model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2,
        #                     return_sequences=True, input_shape=(lookback, len(self.filters))))
        # self.model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        # self.model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))

        # self.model.add(Dense(units=16))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mse')

        earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                                patience=25, restore_best_weights=True)

        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs,
                                      validation_data=(self.X_Test, self.y_test),
                                      shuffle=False, callbacks=[earlystopping])


class GRU_MODEL(Model):
    def __init__(self, df, filters):
        super().__init__(df, filters=filters)

    def process_data(self, lookback, train_size, scaler=StandardScaler):
        data = self.data.copy()

        train_data, test_data = train_test_split(data, test_size=1 - train_size, shuffle=False)

        self.scaler = scaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        X_train, y_train = [], []
        for i in range(len(train_data) - lookback - 1):
            X_train.append(train_data[i:i + lookback])
            y_train.append(train_data[i + lookback, 0])
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)

        # Process the testing data
        X_test, y_test = [], []
        for i in range(len(test_data) - lookback - 1):
            X_test.append(test_data[i:i + lookback])
            y_test.append(test_data[i + lookback, 0])
        self.X_Test, self.y_test = np.array(X_test), np.array(y_test)
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], len(self.filters)))
        self.X_Test = self.X_Test.reshape((self.X_Test.shape[0], self.X_Test.shape[1], len(self.filters)))

    def fit(self, lookback=10, train_size=0.8, scaler=StandardScaler, epochs=300):
        self.process_data(lookback, train_size, scaler)
        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        self.model = Sequential()
        self.model.add(GRU(units=256, input_shape=(lookback, len(self.filters))))
        # self.model.add(GRU(units=128, dropout=0.2, recurrent_dropout=0.2,
        #                     return_sequences=True, input_shape=(lookback, len(self.filters))))
        # self.model.add(GRU(units=64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        # self.model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))

        # self.model.add(Dense(units=16))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                                patience=25, restore_best_weights=True)

        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs,
                                      validation_data=(self.X_Test, self.y_test),
                                      shuffle=False, callbacks=[earlystopping])


print(analyzers[-3].ticker)
analyzers[-3].df.head(100)

# raise ValueError()

# import plotly.graph_objs as go
#
# trace = go.Scatter(x=analyzers[0].df['Date'], y=analyzers[0].df['Price'], name='Price')
#
# layout = go.Layout(title='Price Graph', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
#
# fig = go.Figure(data=[trace], layout=layout)
#
# fig.show()


class KNNModel(Model):
    # def process_data(self, lookback, train_size, scaler=StandardScaler):
    #     # data = self.data.copy()

    #     train_data, test_data = train_test_split(self.data, test_size=1 - train_size, shuffle=False)

    #     # self.scaler = MinMaxScaler()
    #     # self.scaler = StandardScaler()
    #     self.scaler = scaler()
    #     self.scaler.fit(train_data)
    #     train_data = self.scaler.transform(train_data)
    #     test_data = self.scaler.transform(test_data)
    #     X_train, y_train = [], []
    #     for i in range(len(train_data) - lookback - 1):
    #         X_train.append(train_data[i:i + lookback])
    #         y_train.append(train_data[i + lookback, 0])
    #     self.X_train, self.y_train = np.array(X_train), np.array(y_train)

    #     # Process the testing data
    #     X_test, y_test = [], []
    #     for i in range(len(test_data) - lookback - 1):
    #         X_test.append(test_data[i:i + lookback])
    #         y_test.append(test_data[i + lookback, 0])
    #     self.X_Test, self.y_test = np.array(X_test), np.array(y_test)
    #     self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], len(self.filters)))
    #     print(f"self.X_Test shape: {self.X_Test.shape}")
    #     print(f"len(self.filters): {len(self.filters)}")
    #     self.X_Test = self.X_Test.reshape((self.X_Test.shape[0], self.X_Test.shape[1], len(self.filters)))
    def process_data(self, lookback, train_size, scaler=StandardScaler):
        train_data, test_data = train_test_split(self.data, test_size=1 - train_size, shuffle=False)

        self.scaler = scaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        X_train, y_train = [], []
        for i in range(len(train_data) - lookback - 1):
            X_train.append(train_data[i:i + lookback])
            y_train.append(train_data[i + lookback, 0])
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)

        X_test, y_test = [], []
        for i in range(len(test_data) - lookback - 1):
            X_test.append(test_data[i:i + lookback])
            y_test.append(test_data[i + lookback, 0])
        self.X_Test, self.y_test = np.array(X_test), np.array(y_test)

        # Reshape X_train and X_Test to have two dimensions
        self.X_train = self.X_train.reshape((self.X_train.shape[0], -1))
        self.X_Test = self.X_Test.reshape((self.X_Test.shape[0], -1))

    # def process_data(self, train_size):
    #     X = self.data[self.filters]
    #     y = self.data[[self.filters[0]]]

    #     self.X_train, self.X_Test, self.y_train, self.y_test = train_test_split(X, y, test_size=1 - train_size,
    #                                                                             shuffle=False)

    def fit(self, train_size=0.8, lookback=10, scaler=StandardScaler, **kwargs):
        self.process_data(train_size=train_size, lookback=lookback, scaler=scaler)
        self.model = KNeighborsRegressor(n_neighbors=5)
        self.model.fit(self.X_train, self.y_train)
        # self.train_predict = self.model.predict(self.X_train)
        # self.test_predict = self.model.predict(self.X_Test)

        # self.train_rmse = math.sqrt(mean_squared_error(self.y_train, self.train_predict))
        # self.train_mae = mean_absolute_error(self.y_train, self.train_predict)
        # print(f"[{self.filters}] [{self.__class__.__name__}] Train RMSE =", self.train_rmse)
        # print(f"[{self.filters}] [{self.__class__.__name__}] Train MAE =", self.train_mae)

        # self.test_rmse = math.sqrt(mean_squared_error(self.y_test, self.test_predict))
        # self.test_mae = mean_absolute_error(self.y_test, self.test_predict)
        # print(f"[{self.filters}] [{self.__class__.__name__}] Test RMSE =", self.test_rmse)
        # print(f"[{self.filters}] [{self.__class__.__name__}] Test MAE =", self.test_mae)

    # def draw_train_graph(self, ticker=None, ax=None):
    #     ax.plot(self.y_train, label="Actual")
    #     ax.plot(self.train_predict, label="Predicted")
    #     ax.legend()
    #     ax.set_title(f"Train Dataset [{self.__class__.__name__}] [{self.filters}] [{ticker}]")

    # def draw_test_graph(self, ticker=None, ax=None):
    #     ax.plot(self.y_test, label="Actual")
    #     ax.plot(self.test_predict, label="Predicted")
    #     ax.legend()
    #     ax.set_title(f"Train Dataset [{self.__class__.__name__}] [{self.filters}] [{ticker}]")


import json

params = [
    ['Price'],
    ['Price', 'TweetsToneAvg'],
    ['Price', 'NewsToneAvg'],
    ['Price', 'TweetsToneAvg', 'NewsToneAvg'],
    ['Price', 'TweetsToneAvg', 'TweetsTonePositive', 'TweetsToneNeutral', 'TweetsToneNegative'],
    ['Price', 'NewsToneAvg', 'NewsTonePositive', 'NewsToneNeutral', 'NewsToneNegative'],
    ['Price', 'TweetsToneAvg', 'TweetsTonePositive', 'TweetsToneNeutral', 'TweetsToneNegative', 'NewsToneAvg',
     'NewsTonePositive', 'NewsToneNeutral', 'NewsToneNegative']
]
results_gru = {}
try:
    with open("results.json", 'r', encoding="utf-8") as f:
        export_data = json.loads(f.read())
except:
    export_data = {}
# export_data = {}
models = [LSTM_MODEL, GRU_MODEL, KNNModel]
# models = [KNNModel]
for analyzer in analyzers:
    results_gru[analyzer.ticker] = {}
    if analyzer.ticker not in export_data.keys():
        export_data[analyzer.ticker] = {}
    for param in params:
        current_param = '_'.join(param)
        results_gru[analyzer.ticker][current_param] = {}

        if current_param not in export_data[analyzer.ticker].keys():
            export_data[analyzer.ticker][current_param] = {}
        results_gru[analyzer.ticker][current_param]['params'] = param[:]
        results_gru[analyzer.ticker][current_param]['models'] = {}
        export_data[analyzer.ticker][current_param]['params'] = param[:]

        if 'models' not in export_data[analyzer.ticker][current_param].keys():
            export_data[analyzer.ticker][current_param]['models'] = {}
        for model in models:
            print(model.__name__)
            results_gru[analyzer.ticker][current_param]['models'][model.__name__] = {
                "model": model(analyzer.df.copy(), param)
            }
            results_gru[analyzer.ticker][current_param]['models'][model.__name__]['model'].fit(
                scaler=MinMaxScaler,
                epochs=300)
            results_gru[analyzer.ticker][current_param]['models'][model.__name__][
                'model'].plot_multiple_graphs(
                analyzer.ticker)
            export_data[analyzer.ticker][current_param]['models'][model.__name__] = {
                'stats': results_gru[analyzer.ticker][current_param]['models'][model.__name__][
                    'model'].export()
            }
            if model.__name__ != "KNNModel":
                results_gru[analyzer.ticker][current_param]['models'][model.__name__][
                    'model'].model.save(f"models/{analyzer.ticker}_{model.__name__}_{current_param}.h5")
            # print(analyzers[_].df)
            # result_lstm[-1].append()
            # result_lstm[-1][-1].fit(scaler=MinMaxScaler, epochs=10)
            # # result[-1][-1].draw_train_graph(analyzer.ticker)
            # # result[-1][-1].draw_test_graph(analyzer.ticker)
            # result_lstm[-1][-1]
    with open(f"results/{analyzer.ticker}.json", 'w', encoding="utf-8") as f:
        data = export_data[analyzer.ticker].copy()
        f.write(json.dumps(export_data))
    results_gru[analyzer.ticker] = {}
# print(export_data)
with open("results.json", 'w', encoding="utf-8") as f:
    data = export_data.copy()
    f.write(json.dumps(export_data))

# raise ValueError()
export_data = {}

# import json
# params = [
#     ['Price'],
#     ['Price', 'TweetsToneAvg'],
#     ['Price', 'NewsToneAvg'],
#     ['Price', 'TweetsToneAvg', 'NewsToneAvg'],
#     ['Price', 'TweetsToneAvg', 'TweetsTonePositive', 'TweetsToneNeutral', 'TweetsToneNegative'],
#     ['Price', 'NewsToneAvg', 'NewsTonePositive', 'NewsToneNeutral', 'NewsToneNegative'],
#     ['Price', 'TweetsToneAvg', 'TweetsTonePositive', 'TweetsToneNeutral', 'TweetsToneNegative', 'NewsToneAvg',
#      'NewsTonePositive', 'NewsToneNeutral', 'NewsToneNegative']
# ]
# results_gru = {}
# export_data = {}
# models = [LSTM_MODEL]
# for analyzer in analyzers:
#     results_gru[analyzer.ticker] = {}
#     export_data[analyzer.ticker] = {}
#     for param in params:
#         current_param = '_'.join(param)
#         results_gru[analyzer.ticker][current_param] = {}
#         export_data[analyzer.ticker][current_param] = {}
#         results_gru[analyzer.ticker][current_param]['params'] = param[:]
#         results_gru[analyzer.ticker][current_param]['models'] = {}
#         export_data[analyzer.ticker][current_param]['params'] = param[:]
#         export_data[analyzer.ticker][current_param]['models'] = {}
#         for model in models:
#             results_gru[analyzer.ticker][current_param]['models'][model.__class__.__name__] = {
#                 "model": model(analyzer.df.copy(), param)
#             }
#             results_gru[analyzer.ticker][current_param]['models'][model.__class__.__name__]['model'].fit(scaler=MinMaxScaler,
#                                                                                            epochs=1)
#             results_gru[analyzer.ticker][current_param]['models'][model.__class__.__name__]['model'].plot_multiple_graphs(
#                 analyzer.ticker)
#             export_data[analyzer.ticker][current_param]['models'][model.__class__.__name__] = {
#                 'stats': results_gru[analyzer.ticker][current_param]['models'][model.__class__.__name__]['model'].export()
#             }

#             # print(analyzers[_].df)
#             # result_lstm[-1].append()
#             # result_lstm[-1][-1].fit(scaler=MinMaxScaler, epochs=10)
#             # # result[-1][-1].draw_train_graph(analyzer.ticker)
#             # # result[-1][-1].draw_test_graph(analyzer.ticker)
#             # result_lstm[-1][-1]
# print(export_data)
# with open("results_gru.json", 'w', encoding="utf-8") as f:
#     data = export_data.copy()
#     f.write(json.dumps(export_data))

export_data = {}

# print("a")
# with open
#
# for ticker, data_ticker in results_gru.items():
#     for param, data_param in data_ticker.items():
#         for model, data_model in data_param['models'].items():
#             data_model.model.save(f'{ticker}_{model}_{param}.h5')
#
