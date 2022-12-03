# MuskTweetsToTeslaStock

Predict Tesla stock price using Elon Musk Tweets.

By using previous day Tesla Stock close price 
and Elon Musk Tweets info such as number of tweets, number of likes and tweets polarity it trys 
to predict next Tesla Stock close price.

# Installation
`pip install -r requirements.txt`

# Adjustments
- Change prediction timeframe:
```     
self.prediction_start_date = '2018-01-02'
self.prediction_end_date = '2022-03-05
```
- Change number of days used for predictions:
```
self.n_future = 1   #Coming day that will be predicted
self.n_past = 14    #Number of past days used for prediction
```

# Datasets
Program uses two datasets from Kaggle:
  - [TESLA Stock Data](https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021)  
  - [Elon Musk Tweets](https://www.kaggle.com/datasets/ayhmrba/elon-musk-tweets-2010-2021)

# Program operations
 - At first program creates Elon Musk Tweets an Tesla Stock Price dataframes from datasets `.csv` files.
 - Then it gets each Tweet sentiment using `TextBlob` library and adds it to the dataframe.
 - Every Tweet after market close (4pm) is moved to next day.
 - Every Tweet from not business day is moved to next business day.
 - Next it adds Tesla Stock close value for each date with previous day close value as well.
 - Training dataframe for our model with all data:
 ![training_df](https://user-images.githubusercontent.com/117664884/205384640-573a553b-a3ce-40c0-91a4-70231144a2c0.PNG)
 - Data is reshaped and fited to Keras model.
 - Learning time
 - Predicted Tesla Stock close values are ready.

# Results
- Diagram showing final output. As you can see program is quite accurate, original and predicted values are pretty close.
![diagram](https://user-images.githubusercontent.com/117664884/205378666-34d14a9a-c2ba-417d-804d-cba7b23b0166.PNG)
- Final dataframe with predicted close values and with prediction error informations.
![final_df](https://user-images.githubusercontent.com/117664884/205379775-d67f7b06-cb24-4103-ba75-e1ac8355b958.PNG)
