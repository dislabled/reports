#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
A module for manipulating and visualizing sensor data.

This module includes classes for data manipulation (DataManipulator) and plotting (Plotter)
based on Pandas DataFrame containing sensor data.

Classes:
    - DataManipulator: Performs data manipulation tasks like estimating threshold reach time, computing slope, etc.
    - Plotter: Generates plots for specified sensor data against time.

Usage:
    1. Create an instance of DataParser from the 'dataparser' module to parse sensor data from a file.
    2. Pass the parsed data to DataManipulator or Plotter to perform data manipulations or generate plots, respectively.

Example:
    from dataparser import DataParser
    from datamanipulator import DataManipulator
    from plotter import Plotter
    from pathlib import Path

    # Parse the sensor data from a file using DataParser
    parser = DataParser(Path("sensor_data.tsv"))
    parsed_data = parser.parse_file()

    # Perform data manipulations using DataManipulator
    manipulator = DataManipulator(parsed_data)
    starting_point = manipulator.get_starting_point("sensor_name")
    estimated_time = manipulator.estimate_threshold_reach_time("sensor_name", 10, 60, starting_point)

    # Generate plots using Plotter
    plotter = Plotter(parsed_data)
    plotter.gen_plot("sensor_name", threshold_low=20, threshold_high=60, from_starting_point=True)

Author: Stian Knudsen
Date: 2024-01-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import argrelmax
from dataparser import DataParser


class DataManipulator:
    """
    Initialize DataManipulator with the input DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame containing sensor data.
    """

    def __init__(self, data: pd.DataFrame):
        """Initialize the class."""
        self.data = data

    def get_resolution(self) -> pd.Timedelta:
        """Get the resolution of the data."""
        return self.data.index[1] - self.data.index[0]

    def estimate_threshold_reach_time(
        self,
        sensor_name: str,
        threshold: float,
        lookahead_minutes: int | pd.Timedelta = 60,
        starting_point: pd.DatetimeIndex | None = None,
    ) -> pd.Timestamp | None:
        """
        Estimate the time when a sensor might reach a specified threshold.

        Args:
            threshold (float): Threshold value.
            lookahead_minutes (int): Lookahead time in minutes for estimation (default: 60).

        Returns:
            pd.Timestamp: Estimated timestamp when the threshold might be reached.
        """
        # Get the sensor data column.
        sensor_data = self.data[sensor_name]

        if starting_point is None:
            # Use the first timestamp as the starting point.
            starting_point = pd.to_datetime(self.data.index[0])

        print(starting_point)
        # Check if there is enough data to estimate the threshold reach time.
        lookahead_minutes = pd.to_timedelta(lookahead_minutes, unit="minutes")
        entries = int(-abs(lookahead_minutes / self.get_resolution()))
        if self.data.index[entries] < starting_point:
            raise ValueError("Not enough data to estimate the threshold reach time.")

        # Calculate the average rate of change over the last 'lookahead_minutes' minutes.
        recent_data = sensor_data.loc[starting_point:]
        average_change_rate = (recent_data.iloc[-1] - recent_data.iloc[0]) / (lookahead_minutes.total_seconds() / 60)

        # Estimate the number of minutes until the threshold might be reached based on the average rate of change.
        minutes_to_threshold = (threshold - recent_data.iloc[-1]) / average_change_rate

        # Extract the scalar value from the Series.
        minutes_to_threshold_scalar = minutes_to_threshold.iloc[0]

        # Add the estimated minutes to the last timestamp in the data to get the estimated time.
        delta = pd.Timedelta(minutes=minutes_to_threshold_scalar)
        if isinstance(delta, pd.Timedelta):
            return starting_point + delta
        return None

    def get_starting_point(self, sensor_name: str) -> pd.Timestamp | None:
        """
        Get the starting point (last local maximum) timestamp in the data.

        Args:
            sensor_name (str): Name of the sensor.

        Returns:
            pd.Timestamp | None: Timestamp of the last local maximum in the sensor's data or None if not found.
        """
        indices_max = argrelmax(self.data[sensor_name].values, order=5)[0]

        if len(indices_max) > 0:
            # Get the index of the last peak
            starting_point = self.data.index[indices_max[-1]]
            if isinstance(starting_point, pd.Timestamp):
                return starting_point
        return None

    def compute_slope_to_threshold(self, sensor_name: str, threshold: float, lookahead_minutes: int = 60) -> None:
        """
        Compute the slope to a specified threshold and append estimated values to the DataFrame.

        Args:
            sensor_name (str): Name of the sensor.
            threshold (float): Threshold value.
            lookahead_minutes (int): Lookahead time in minutes for estimation (default: 60).

        Returns:
            None
        """
        # Get the sensor data column.
        sensor_data = self.data[sensor_name]

        # Get the starting index of the data.
        starting_point = self.get_starting_point(sensor_name)

        if starting_point is None:
            return None  # Return None if there's no valid starting point.

        # Calculate average change rate over the last 'lookahead_minutes' minutes.
        recent_data = sensor_data.loc[starting_point:]
        average_change_rate = (recent_data.iloc[-1] - recent_data.iloc[0]) / lookahead_minutes
        print(average_change_rate)

        # Estimate the number of data points until the threshold might be reached.
        if average_change_rate != 0:
            data_points_to_threshold = (threshold - recent_data.iloc[-1]) / average_change_rate
        else:
            data_points_to_threshold = np.nan

        # Calculate the time difference between each data point.
        time_diff = (recent_data.index[-1] - recent_data.index[-2]).total_seconds() / 60

        # Create new timestamps based on the time difference and the number of data points.
        new_time_indices = recent_data.index[-1] + pd.to_timedelta(
            np.arange(1, data_points_to_threshold + 1) * time_diff, unit="m"
        )

        # Create new sensor values based on the slope and timestamps.
        new_sensor_values = recent_data.iloc[-1] + average_change_rate * np.arange(1, data_points_to_threshold + 1)

        # Create new DataFrame with the additional data.
        new_data = pd.DataFrame({"Time": new_time_indices, sensor_name: new_sensor_values})

        # Append new data to the existing DataFrame.
        self.data = pd.concat([self.data, new_data], ignore_index=True)

        return new_time_indices[-1]  # Return the estimated time of reaching the threshold.


class Plotter:
    """
    A class to generate plots for specified sensor data against time.

    Args:
        data (pd.DataFrame): Input DataFrame containing sensor data.
    """

    def __init__(self, data: pd.DataFrame):
        """Initialize the class."""
        self.data = data

    def gen_plot(
        self,
        sensor_name: str,
        threshold_low: int | None = None,
        threshold_high: int | None = None,
        from_starting_point: bool = False,
    ) -> None:
        """
        Generate a plot for a specified sensor's data against time.

        Args:
            sensor_name (str): Name of the sensor.
            threshold_low (int | None): Low threshold value for horizontal line (default: None).
            threshold_high (int | None): High threshold value for horizontal line (default: None).
            from_starting_point (bool): Determine if the plot starts from the starting point (default: False).

        Returns:
            None
        """
        start_index = None
        # Check if index is all or from starting_point.
        if from_starting_point:
            manipulator = DataManipulator(self.data)
            start_index = manipulator.get_starting_point(sensor_name)

        if start_index is None:
            plot_data = self.data[sensor_name]
        else:
            plot_data = self.data.loc[start_index:, sensor_name]

        ax = plot_data.plot()
        # Get the unit for the sensor.
        unit = self.data.columns[self.data.columns.get_level_values(0) == sensor_name].get_level_values(2)[0]

        # Modifying the y-axis ticks to append the unit.
        ticks = ax.get_yticks()
        ax.set_yticks(ticks, minor=True)
        ax.set_yticklabels([f"{tick} {unit}" for tick in ticks])

        # Formatting the plot.
        plt.xlabel("")
        plt.ylabel("")
        plt.ylim(10.0, 100.0)
        if threshold_low:
            plt.axhline(y=threshold_low, color="r", linestyle="--", label="Threshold low")
        if threshold_high:
            plt.axhline(y=threshold_high, color="r", linestyle="--", label="Threshold high")
        plt.title(f"Sensor {sensor_name}")

        # Show the plot.
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = DataParser(Path("levels_big.tsv"))
    parsed_data = parser.parse_file()
    manipulator = DataManipulator(parsed_data)
    # print(manipulator.get_starting_point("2HTJ10CW001"))
    start_time = manipulator.get_starting_point("2HTJ10CW001")
    # print(manipulator.estimate_threshold_reach_time("2HTJ10CW001", 10, 60, start_time))
    # manipulator.compute_slope_to_threshold("2HTJ10CW001", 10, 10)

    plotter = Plotter(parsed_data)
    # plotter.gen_plot("2HTJ10CW001", threshold_low=20, threshold_high=60, from_starting_point=False)
    # plotter.gen_plot("2HTJ10CW001", threshold_low=20, threshold_high=60, from_starting_point=True)
