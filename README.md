# London Power Consumption Prediction
A personal attempt to build a RNN(LSTM) model which predicts London's total power consumption given previous consumptions and weathers.
The project is based on Kaggle London Smart Meter dataset, in which only parts of weather and daily power datasets are used.
dataset link: https://www.kaggle.com/jeanmidev/smart-meters-in-london

***Tools Used***:

***Spark***: reduce electricity data from different stations to the sum of all stations to dates and merge it with daily weather

***TensorFlow Keras***: data modeling and prediction

***Sklearn***: PCA transformation
