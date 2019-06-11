# Spark-LSTM-Prediction
A RNN(LSTM) model that predicts total power consumption of London given previous consumptions and weathers.
The project is based on Kaggle London Smart Meter dataset, in which only part of weather and daily power dataset is used.
dataset download: https://www.kaggle.com/jeanmidev/smart-meters-in-london

***Apache Spark***: reduce electricity data from different stations to the sum of all stations on a single day, and then merged with daily weather
***TensorFlow Keras***:data modeling and prediction

The outcome is given in unit of 1k. Please multiply 1k with predicted outcome to get the actual sum of power consumption on the day.

***Note: Red line represents actual data while blue line represents predicted data***

***Training data:***
![alt text](https://github.com/JeffreyW0w/Spark-LSTM-Prediction/blob/master/result_pics/train.png?raw=true)

***Validation data:***
![alt text](https://github.com/JeffreyW0w/Spark-LSTM-Prediction/blob/master/result_pics/valid.png?raw=true)

***Unseen data(test):***
![alt text](https://github.com/JeffreyW0w/Spark-LSTM-Prediction/blob/master/result_pics/test.png?raw=true)


***Conclusion:
There is a strong correlation between weather and total power consumption. This model is able to predict total power consumption 1 day ahead accurately.***
