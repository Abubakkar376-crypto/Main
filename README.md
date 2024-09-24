# Main 
import yfinance as yf

# Download stock data for Google (GOOG) from Yahoo Finance
stock_data = yf.download('GOOG', start='2010-01-01', end='2023-09-01')
print(stock_data.head())
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[['Close']].values)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)
from sklearn.metrics import mean_squared_error
import numpy as np

predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'Root Mean Squared Error: {rmse}')
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(stock_data['Close'], order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction=predicted_stock_price)

if __name__ == "__main__":
    app.run(debug=True)
