import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


if __name__ == '__main__':
    # fix random seed for reproducibility
    np.random.seed(7)

    ##############################
    # READ DATA
    ##############################

    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv("data_BTC.csv", usecols=[1, 2, 3], parse_dates=['timestamp'], date_parser=dateparse)
    df = df.reset_index().drop_duplicates(subset=['timestamp'], keep='last').set_index('timestamp').sort_index()

    ##############################
    # DATA PREP
    ##############################

    # avgprice
    df['avgPrice'] = df[["lowestPrice", "highestPrice"]].mean(axis=1)
    df = df[['avgPrice']]

    # interpolate missing values
    df = df.resample('Min').interpolate(method='time')

    df.plot()
    plt.show()

    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)

    # split into train / cross val / test
    train_size = int(len(df) * 0.95)
    test_size = len(df) - train_size
    train, test = df[0:train_size, :], df[train_size: len(df), :]

    print("using %i sample to train and %i for test" % (train_size, test_size))

    # reshape into X=t and Y=t+1
    look_back = 10
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    #####################
    # Learn Bitch
    #####################
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=40, batch_size=1, verbose=2)

    #####################
    # Predict Bitch
    #####################
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % trainScore)
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % testScore)

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(df)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(df) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(df))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

    #save model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("weights.h5")

    #accuracy:
    score = model.evaluate(testX, testPredict)
    print(score)
