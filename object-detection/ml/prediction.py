#!/usr/bin/env python
import logging
from collections import defaultdict

import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# setup the logger
logger = logging.getLogger(__name__)


def create_sequences(data, window):
    logger.debug(f"length of data: {len(data)}")
    x, y = [], []
    for index in range(len(data) - window):
        x.append(data[index:index + window])
        y.append(data[index + window])
    return np.array(x), np.array(y)


def build_model(input_shape):
    """build the LSTM model with 2 input, dense and dropout layers each"""
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=input_shape[1], activation="linear"))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def predict(model, actual_data, step_window=15, time_window=15) -> pd.DataFrame:
    """function to make predictions for the next 15 minutes"""
    predictions = []
    actual_data_ndarray = actual_data.to_numpy()
    actual_data_ndarray = actual_data_ndarray[-step_window:]
    for _ in range(time_window):
        prediction = model.predict(np.array([actual_data_ndarray]))
        predictions.append(prediction[0])
        actual_data_ndarray = np.vstack([actual_data_ndarray[1:], prediction])
    predicted_dataframe = pd.DataFrame(predictions, columns=actual_data.columns)
    predicted_dataframe = predicted_dataframe.applymap(lambda count: max(0, int(count)))
    print(predicted_dataframe)
    return predicted_dataframe.sum()


def run_predictions(object_detection_df: pd.DataFrame):

    # set the index to timestamp
    object_detection_df.index = pd.to_datetime(object_detection_df["Timestamp"])

    # find all the unique turning patterns
    turning_patterns = object_detection_df["Turning Pattern"].unique()

    # Define the window sequence size
    window_size = 15

    # datastore for storing predictions
    predicted_datastore = defaultdict(lambda: dict())

    for turning_pattern in turning_patterns:

        # filter the detected vehicle data for target turning pattern
        turning_pattern_dataframe = object_detection_df[object_detection_df["Turning Pattern"] == turning_pattern]

        # training_dataset = turning_pattern_dataframe[:15]
        # prediction_dataset = turning_pattern_dataframe[15:]

        target_vehicle_class = ["Bicycle", "Bus", "Cars", "Two-Wheeler", "Three-Wheeler", "LCV", "Truck"]

        # for vehicle_class in target_vehicle_class:

        # vehicle_training_dataset = training_dataset[[vehicle_class]]
        # vehicle_evaluation_dataset = prediction_dataset[[vehicle_class]]
        vehicle_turning_pattern_dataframe = turning_pattern_dataframe[target_vehicle_class]

        # Create sequences for both directions
        x_sequence, y_sequence = create_sequences(vehicle_turning_pattern_dataframe.to_numpy(), window_size)

        # build the lstm model
        model = build_model((x_sequence.shape[1], x_sequence.shape[2]))

        # mo the model summary
        logger.info(model.summary())

        # train the model
        model.fit(x_sequence, y_sequence, epochs=128, batch_size=32)

        # prediction of vehicle count for next 30 minutes
        predicted_vehicle_count = predict(model, vehicle_turning_pattern_dataframe, step_window=window_size, time_window=30)

        # add the prediction to local datastore
        predicted_datastore[turning_pattern] = predicted_vehicle_count.to_dict()
    logger.info(f"prediction data: {predicted_datastore}")
    return predicted_datastore.items()


if __name__ == "__main__":
    data_path = "merged_file.csv"
    logger.debug(f"reading data from file path: {data_path}")
    data = pd.read_csv(data_path)
    data.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    run_predictions(data)
