#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import argrelmax


class DataParser:
    def __init__(self, filename: Path) -> None:
        """Initialize the class."""
        self.filename = filename

    def parse_file(self) -> pd.DataFrame:
        """
        Parse the input file and format the data.

        Args:
            filename (Path): Path to the file to be parsed.

        Returns:
            pd.DataFrame: Parsed data in a formatted DataFrame.
        """
        # Open the input file in read mode.
        with open(self.filename, "r") as file:
            # Read the file and store its contents in a list.
            lines = file.readlines()

        # Remove newlines.
        while "\n" in lines:
            lines.remove("\n")

        # Remove header and footer.
        lines = lines[3:-4]

        # Remove all instances of ":av".
        lines = [line.replace(":av", "") for line in lines]

        # Fix missing headers.
        lines[1] = "Time of sensors\t" + lines[1]
        lines[2] = "localtime\t" + lines[2]

        # Merge lines that have AM/PM separated, Valmet generation bug.
        cleaned_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Check if the line ends with AM or PM
            if line.endswith(("AM", "PM")):
                # Merge the current line with the next line
                line += " " + lines.pop(i + 1).strip()
            cleaned_lines.append(line)
            i += 1

        # Split on tabs and remove newlines.
        return_list = [line.strip().split("\t") for line in cleaned_lines]

        # Make tuples for the header
        column_tuples = list(zip(return_list[0], return_list[1], return_list[2]))

        # Make the DataFrame
        df = pd.DataFrame(return_list[3:])
        # Set the header with the tuples
        df.columns = pd.MultiIndex.from_tuples(column_tuples)

        # Extract the datetime string and AM/PM separately
        datetime_str = df.iloc[:, 0]
        am_pm = datetime_str.str.extract(r"(\b(?:AM|PM)\b)", expand=False)
        datetime_str = datetime_str.str.replace(r"\b(?:AM|PM)\b", "", regex=True).str.strip()

        # Combine date and time and convert to datetime
        datetime_combined = pd.to_datetime(datetime_str + " " + am_pm, format="%m/%d/%Y %I:%M:%S %p")

        # Update the first column with the merged datetime
        df.iloc[:, 0] = datetime_combined

        # Change the datatypes of the columns
        df[column_tuples[1:]] = df[column_tuples[1:]].astype(float)

        return pd.DataFrame(df)


class DataManipulator:
    def __init__(self, data: pd.DataFrame):
        """Initialize the class."""
        self.data = data

    def estimate_threshold_reach_time(
        self, sensor_name: str, threshold: float, lookahead_minutes: int = 60
    ) -> pd.Timestamp | None:
        """
        Estimate the time when a sensor might reach a specified threshold.

        Args:
            sensor_name (str): Name of the sensor.
            threshold (float): Threshold value.
            lookahead_minutes (int): Lookahead time in minutes for estimation (default: 60).

        Returns:
            pd.Timestamp: Estimated timestamp when the threshold might be reached.
        """
        # Get the sensor data column
        sensor_data = self.data[sensor_name]

        # Get the starting index of the data
        starting_point = self.get_starting_point(sensor_name)

        if starting_point is None:
            return None

        # Get the "Time" column as a pandas Series
        time_column = self.data["Time"]
        starting_time = time_column.loc[starting_point]

        # Calculate the average rate of change over the last 'lookahead_minutes' minutes
        recent_data = sensor_data.loc[starting_point:]
        average_change_rate = (recent_data.iloc[-1] - recent_data.iloc[0]) / lookahead_minutes

        # Estimate the number of minutes until the threshold might be reached based on the average rate of change
        minutes_to_threshold = (threshold - recent_data.iloc[-1]) / average_change_rate

        # Extract the scalar value from the Series
        minutes_to_threshold_scalar = minutes_to_threshold.iloc[0]

        # Add the estimated minutes to the last timestamp in the data to get the estimated time
        estimated_time = starting_time + pd.Timedelta(minutes=minutes_to_threshold_scalar)

        return estimated_time

    def get_starting_point(self, sensor_name: str) -> pd.Index | None:
        """Get the starting point of the data."""
        indices_max = argrelmax(self.data[sensor_name].values, order=5)[0]

        if len(indices_max) > 0:
            # Get the index of the last peak
            return indices_max[-1]
        else:
            return None

    def add_slope_column(self, sensor_name: str, threshold: float, lookahead_minutes: int = 60) -> None:
        """
        Calculate the slope between the last recorded value and the estimated threshold reach,
        and add the slope values as a new column in the DataFrame.

        Args:
            sensor_name (str): Name of the sensor.
            threshold (float): Threshold value.
            lookahead_minutes (int): Lookahead time in minutes for estimation (default: 60).

        Returns:
            None
        """
        recent_data = self.data[sensor_name].tail(lookahead_minutes)

        # Calculate the average rate of change over the last 'lookahead_minutes' minutes
        average_change_rate = (recent_data.iloc[-1] - recent_data.iloc[0]) / lookahead_minutes

        # Estimate the number of minutes until the threshold might be reached based on the average rate of change
        minutes_to_threshold = (threshold - recent_data.iloc[-1]) / average_change_rate

        # Extract the scalar value from the Series
        minutes_to_threshold_scalar = minutes_to_threshold.iloc[0]

        # Find the estimated timestamp when the threshold might be reached
        estimated_time = self.data.index[-1] + pd.Timedelta(minutes=minutes_to_threshold_scalar)

        # Get the values for the estimated timestamp and the last recorded timestamp
        last_value = self.data[sensor_name].iloc[-1]
        estimated_threshold_value = self.data[sensor_name].loc[estimated_time]

        # Calculate the slope between the last recorded value and the estimated threshold reach
        slope = (estimated_threshold_value - last_value) / minutes_to_threshold_scalar

        # Create a new column with the calculated slope values
        self.data["Slope"] = pd.Series([slope] * len(self.data), index=self.data.index)


class Plotter:
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
        # Get the Time header tuple, and the starting point before index changes.
        time_tuple = self.data.columns[self.data.columns.get_level_values(0) == "Time"][0]

        start_index = None
        if from_starting_point:
            manipulator = DataManipulator(self.data)
            start_index = manipulator.get_starting_point(sensor_name)
            start_index = str(np.datetime_as_string(manipulator.data.loc[start_index]["Time"])[0])

        # Convert index to datetime for time representation.
        self.data.set_index(time_tuple, inplace=True)

        if start_index is None:
            plot_data = self.data[sensor_name]
        else:
            plot_data = self.data.loc[start_index:, sensor_name]

        ax = plot_data.plot()
        # Get the unit for the sensor.
        unit = self.data.columns[self.data.columns.get_level_values(0) == sensor_name].get_level_values(2)[0]

        # Modifying the y-axis ticks to append the unit.
        ticks = ax.get_yticks()
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
    manipulator.add_slope_column("2HTJ10CW001", 10, 3600)

    plotter = Plotter(parsed_data)
    plotter.gen_plot("2HTJ10CW001", threshold_low=20, threshold_high=60, from_starting_point=True)
