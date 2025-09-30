# Ex.No: 6 HOLT WINTERS METHOD
### Date: 30.9.2025
### AIM: 
To implement the Holt Winters Method Model using Python.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative
trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and
Evaluate the model predictions against test data
6. Create teh final model and predict future data and plot it

### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv("gold.csv", parse_dates=['Date'], index_col='Date')
print(data.head())

data_monthly = data['Close'].resample('MS').mean()
print(data_monthly.head())
data_monthly.plot(title="Monthly Prices")
plt.show()

scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)
scaled_data.plot(title="Scaled Monthly Gold Closing Prices",)
plt.show()

decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()

scaled_data=scaled_data+1
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()

test_predictions = model.forecast(steps=len(test_data))
ax = train_data.plot(label="Train")
test_data.plot(ax=ax, label="Test")
test_predictions.plot(ax=ax, label="Forecast")
ax.set_title("Gold Price Forecasting - Holt Winters Method")
ax.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
print("Test RMSE:", rmse)

final_model = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
future_forecast = final_model.forecast(steps=12)
ax = scaled_data.plot(label="Historical")
future_forecast.plot(ax=ax, label="Future Forecast")
ax.set_title("Future Gold Price Forecast (Scaled)")
ax.legend()
plt.show()
```
### OUTPUT:
Scaled_data plot:
<img width="697" height="547" alt="1 sacled plot" src="https://github.com/user-attachments/assets/b135e98a-582b-44b1-84b3-3fdcbf114ab6" />

Decomposed plot:
<img width="817" height="581" alt="2 decomposition" src="https://github.com/user-attachments/assets/5141b7b1-4992-41da-8918-dc30c6fea5fc" />

Test prediction:
<img width="728" height="565" alt="3 test prediction" src="https://github.com/user-attachments/assets/000bb9b4-bdce-40a6-8825-5ee6b3212137" />

RMSE,Standard Deviation and Mean:
<img width="391" height="76" alt="4 calculations" src="https://github.com/user-attachments/assets/d0a45a2b-133a-47b7-a584-d90a50483568" />

Final Predciton:
<img width="736" height="558" alt="5 final prediction" src="https://github.com/user-attachments/assets/60eba383-bb78-48f3-a497-f928db1cee0b" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
