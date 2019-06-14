# LSTM Power Consumption Prediction
A personal attempt to build a RNN(LSTM) model that predicts total power consumption of London given previous consumptions and weathers.
The project is based on Kaggle London Smart Meter dataset, in which only part of weather and daily power dataset is used.
dataset download: https://www.kaggle.com/jeanmidev/smart-meters-in-london

***Apache Spark***: reduce electricity data from different stations to the sum of all stations to dates, and then merge it with daily weather

***TensorFlow Keras***: data modeling and prediction

The outcome is given in unit of 1k. Please multiply 1k with predicted outcome to get the actual sum of power consumption on the day.
In conclusion, the relationship between weather data and total power consumption is not strong. The prediction is not very accurate. More well rounded datasets would improve the result.
