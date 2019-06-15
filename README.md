# London Power Consumption Prediction
A personal attempt to build a RNN(LSTM) model that predicts total power consumption of London given previous consumptions and weathers.
The project is based on Kaggle London Smart Meter dataset, in which only part of weather and daily power dataset is used.
dataset download: https://www.kaggle.com/jeanmidev/smart-meters-in-london

***Apache Spark***: reduce electricity data from different stations to the sum of all stations to dates, and then merge it with daily weather

***TensorFlow Keras***: data modeling and prediction

***Sklearn***: PCA transformation

In conclusion, the relationship between weather data and total power consumption is not strong. Pure LSTM result is disappointing. After applying PCA transformation, with high dimensionality, a long-term trend of electricity usage could be predicted, though it is not highly accurate in terms of days. Adding more well rounded datasets, other than weather, would potentially improve the result.
