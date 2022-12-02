import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
from textblob import TextBlob
import re
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

class TweetsVsStockPrice:

    def __init__(self):
        self.start_date = '2011-12-01'
        self.end_date = '2022-03-05'

        self.prediction_start_date = '2018-01-02'
        self.prediction_end_date = '2022-03-05'

        self.n_future = 1
        self.n_past = 14

    def str_to_date(self, str):
        split = str.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])

        return datetime.date(year=year, month=month, day=day)

    def clean_tweet(self, tweet):
        tweet = tweet.lower()
        tweet = re.sub("@[A-Za-z0-9_]+","", tweet)
        tweet = re.sub("#[A-Za-z0-9_]+","", tweet)
        tweet = re.sub(r"http\S+", "", tweet)
        tweet = re.sub(r"www.\S+", "", tweet)

        emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    "]+", flags=re.UNICODE)
        tweet = emoji_pattern.sub(r'', tweet)

        return tweet

    def create_file_list(self):
        files = []
        current_directory = os.getcwd()
        self.folder_directory = os.path.join(current_directory, 'tweets')
        for file in os.listdir(self.folder_directory):
            files.append(file)

        return files

    def get_tweet_time(self):
        tweets_time = self.year_tweets_df['date'].to_list()
        for i in range(len(tweets_time)):
            tweets_time[i] = tweets_time[i].split()[1]
        self.year_tweets_df['time'] = tweets_time

    def change_date_format(self, str):
        date = str.split()[0]

        return date

    def add_stock_time(self, dif):
        tweets_time = self.year_tweets_df['time'].to_list()
        for i in range(len(tweets_time)):
            hour = tweets_time[i].split(':')[0]
            hour = int(hour)
            hour = hour - dif
            tweets_time[i] = hour
        tweets_date = self.year_tweets_df['date'].to_list()
        for i in range(len(tweets_date)):
            if tweets_time[i] > 16:
                tweets_date[i] = tweets_date[i] + datetime.timedelta(days=1)
        self.year_tweets_df['stock_time'] = tweets_date

    def change_df_columns_names(self):
        self.year_tweets_df.rename(columns={'replies_count': 'nreplies', 'retweets_count':'nretweets', 'likes_count': 'nlikes'}, inplace=True)
        rep = self.year_tweets_df['nreplies']
        ret = self.year_tweets_df['nretweets']
        lik = self.year_tweets_df['nlikes']
        self.year_tweets_df['nreplies'] = lik
        self.year_tweets_df['nretweets'] = rep
        self.year_tweets_df['nlikes'] = ret

    def tweets_polarity(self):
        tweets = self.tweets_df['tweet'].to_list()
        polarity_data = []
        for tweet in tweets:
            tweet = TextBlob(tweet)
            tweet_polarity = tweet.sentiment.polarity
            polarity_data.append(tweet_polarity)
        self.tweets_df['polarity'] = polarity_data

    def tweet_date_to_trading_date(self):
        trading_dates = self.stock_df['Date'].to_list()
        tweets_dates = self.tweets_df['stock_time'].to_list()
        tweets_dates_to_trading_dates = []
        for i in range(len(tweets_dates)):
            date = tweets_dates[i]
            while True:
                if date in trading_dates:
                    tweets_dates_to_trading_dates.append(date)
                    break
                else:
                    date = date + datetime.timedelta(days=1)
        self.tweets_df['trading_date'] = tweets_dates_to_trading_dates

    def strip_df_by_date(self, start_date, end_date, df):
        start_date = TSP.str_to_date(start_date)
        end_date = TSP.str_to_date(end_date)
        days_between = []
        delta = end_date - start_date
        for i in range(delta.days + 1):
            day = start_date + datetime.timedelta(days=i)
            days_between.append(day)
        try:
            df = df[df['date'].isin(days_between)]
        except:
            df = df[df['Date'].isin(days_between)]
        df = df.reset_index(drop=True)

        return df

    def add_yesterdays_dates(self):
        close_values = self.training_df['close'].to_list()
        close_values.pop()
        self.training_df = self.training_df.iloc[1:, :]
        self.training_df['previous_close'] = close_values

    def create_stock_df(self):
        self.stock_df = pd.read_csv('TSLA.csv')
        self.stock_df = self.stock_df[['Date', 'Open', 'Close']].copy()
        self.stock_df['Date'] = self.stock_df['Date'].apply(TSP.str_to_date)

    def create_tweets_df(self):
        files = TSP.create_file_list()
        self.tweets_df = pd.DataFrame()
        for file_name in files:
            file_name = os.path.join(self.folder_directory, file_name)
            self.year_tweets_df = pd.read_csv(file_name)
            self.year_tweets_df = self.year_tweets_df.iloc[::-1]
            if file_name == os.path.join(self.folder_directory, '2020.csv'):
                TSP.get_tweet_time()
                self.year_tweets_df['date'] = self.year_tweets_df['date'].apply(TSP.change_date_format)
                self.year_tweets_df['date'] = self.year_tweets_df['date'].apply(TSP.str_to_date)
                TSP.add_stock_time(5)
            else:
                TSP.change_df_columns_names()
                self.year_tweets_df['date'] = self.year_tweets_df['date'].apply(TSP.str_to_date)
                TSP.add_stock_time(8)
            self.tweets_df = pd.concat([self.tweets_df, self.year_tweets_df])    

        self.tweets_df = self.tweets_df.reset_index(drop=True)
        self.tweets_df['tweet'] = self.tweets_df['tweet'].apply(TSP.clean_tweet)
        TSP.tweets_polarity()
        TSP.create_stock_df()
        TSP.tweet_date_to_trading_date()
        self.tweets_df = self.tweets_df[['date', 'trading_date', 'tweet', 'nlikes', 'nreplies', 'nretweets', 'polarity']].copy() 

    def create_training_df(self):
        self.tweets_df = TSP.strip_df_by_date(self.start_date, self.end_date, self.tweets_df)
        trading_date = self.tweets_df['trading_date'].to_list()
        nlikes = self.tweets_df['nlikes'].to_list()
        nreplies = self.tweets_df['nreplies'].to_list()
        nretweets = self.tweets_df['nretweets'].to_list()
        polarity = self.tweets_df['polarity'].to_list()

        training_date_list = []
        training_ntweets_list = []
        training_nlikes_list = []
        training_nreplies_list = []
        training_nretweets_list = []
        training_polarity_list = []

        previous_date = None

        for i in range(len(trading_date)):
            if trading_date[i] == previous_date:
                training_ntweets += 1
                training_nlikes += nlikes[i]
                training_nreplies += nreplies[i]
                training_nretweets += nretweets[i]
                training_polarity += polarity[i]
            else:
                if previous_date == None:
                    training_nlikes = nlikes[i]
                    training_nreplies = nreplies[i]
                    training_nretweets = nretweets[i]
                    training_polarity = polarity[i]
                    previous_date = trading_date[i]
                    training_ntweets = 1
                else:
                    training_date_list.append(previous_date)
                    training_ntweets_list.append(training_ntweets)
                    training_nlikes_list.append(training_nlikes)
                    training_nreplies_list.append(training_nreplies)
                    training_nretweets_list.append(training_nretweets)
                    training_polarity_list.append(training_polarity/training_ntweets)

                    training_nlikes = nlikes[i]
                    training_nreplies = nreplies[i]
                    training_nretweets = nretweets[i]
                    training_polarity = polarity[i]
                    previous_date = trading_date[i]
                    training_ntweets = 1

        self.training_df = pd.DataFrame()

        self.stock_df = TSP.strip_df_by_date(self.start_date, self.end_date, self.stock_df)
        trading_dates = self.stock_df['Date'].to_list()

        for i in range(len(trading_dates)):
            if not training_date_list[i] == trading_dates[i]:
                training_date_list.insert(i, trading_dates[i])
                training_ntweets_list.insert(i, 0)
                training_nlikes_list.insert(i, 0)
                training_nreplies_list.insert(i, 0)
                training_nretweets_list.insert(i, 0)
                training_polarity_list.insert(i, 0)

        self.training_df['date'] = training_date_list
        self.training_df['ntweets'] = training_ntweets_list
        self.training_df['nlikes'] = training_nlikes_list
        self.training_df['nreplies'] = training_nreplies_list
        self.training_df['nretweets'] = training_nretweets_list
        self.training_df['polarity'] = training_polarity_list

        self.training_df['close'] = self.stock_df['Close'].copy()
        TSP.add_yesterdays_dates()
        self.training_df = self.training_df[['date', 'previous_close', 'ntweets', 'nlikes', 'nreplies', 'nretweets', 'polarity', 'close']]

    def df_scaler(self, df):
        self.scaler = StandardScaler()
        self.scaler = self.scaler.fit(df)
        scaled_df = self.scaler.transform(df)

        return scaled_df

    def reshape_data(self, df, scaled_df, n_future, n_past):
        self.X_train = []
        self.Y_train = []

        for i in range(n_past, len(scaled_df) - n_future + 1):
            self.X_train.append(scaled_df[i - n_past:i, 0:df.shape[1]])
            self.Y_train.append(scaled_df[i + n_future - 1:i + n_future, 0])

        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(64, activation='relu', input_shape=(self.X_train.shape[1], self.X_train.shape[2]), return_sequences=True))
        self.model.add(LSTM(32, activation='relu', return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.Y_train.shape[1]))

        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

        self.model.fit(self.X_train, self.Y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=1)

    def prediction(self, df, df_dates, n_past):

        prediction = self.model.predict(self.X_train)

        prediction_copies = np.repeat(prediction, df.shape[1], axis=-1)
        prediction_values = self.scaler.inverse_transform(prediction_copies)[:,0]

        self.prediction_df = pd.DataFrame()
        self.prediction_df['date'] = df_dates[n_past:]
        self.prediction_df['predicted_close'] = prediction_values
        self.prediction_df = self.prediction_df.reset_index(drop=True)

    def create_final_df(self):
        self.final_df = self.final_df.iloc[self.n_past:, :]
        self.final_df = self.final_df.reset_index(drop=True)
        predicted_close_list = self.prediction_df['predicted_close'].to_list()
        self.final_df['predicted_close'] = predicted_close_list
        close_difference = []
        close_per_difference = []
        close_list = self.final_df['close'].to_list()
        for i in range(len(close_list)):
            close_difference.append(abs(predicted_close_list[i] - close_list[i]))
            close_per_difference.append(abs(predicted_close_list[i] - close_list[i])*100/close_list[i])
        self.final_df['prediction_error'] = close_difference
        self.final_df['prediction_error_%'] = close_per_difference

    def reshape_df(self):
        self.training_df = TSP.strip_df_by_date(self.prediction_start_date, self.prediction_end_date, self.training_df)
        self.training_df = self.training_df.reset_index(drop=True)
        self.final_df = self.training_df.copy()
        self.training_df_dates = self.training_df['date']
        cols = list(self.training_df)[1:8]
        self.training_df = self.training_df[cols].astype(float)

    def main(self):
        TSP.create_tweets_df()
        TSP.create_training_df()
        TSP.reshape_df()
        scaled_training_df = TSP.df_scaler(self.training_df)
        TSP.reshape_data(self.training_df, scaled_training_df, self.n_future, self.n_past)
        TSP.create_model()
        TSP.prediction(self.training_df, self.training_df_dates, self.n_past)
        TSP.create_final_df()
        print(self.final_df)

        plt.plot(self.training_df_dates, self.training_df['close'])
        plt.plot(self.prediction_df['date'], self.prediction_df['predicted_close'])
        plt.legend(['Original values', 'Predicted values'])
        plt.show()



TSP = TweetsVsStockPrice()
TSP.main()