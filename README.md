# Spark-LSTM-Prediction
A RNN(LSTM) model that predicts total power consumption of London given previous consumptions and weathers.
The project is based on Kaggle London Smart Meter dataset, in which only part of weather and daily power dataset is used.
dataset download: https://www.kaggle.com/jeanmidev/smart-meters-in-london
Tools: Apache Spark for data merging and cleaning, TensorFlow Keras for actual model building.

The outcome is given in unit of 1k. Please multiply 1k with predicted outcome to get the actual sum of power consumption on the day.
