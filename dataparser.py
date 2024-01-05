#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
A module to parse input files containing sensor data and format it into a Pandas DataFrame.

This module includes a DataParser class that reads a specific input file, processes the data, and
returns a Pandas DataFrame containing the formatted sensor data.
All based on a badly formatted tsv file from a scada system.

Usage:
    To utilize this module, create an instance of the DataParser class with the file path,
    and use the 'parse_file()' method to obtain the parsed data in a DataFrame.

Example:
    from dataparser import DataParser
    from pathlib import Path

    # Provide the file path containing sensor data
    file_path = Path("path_to_your_file.txt")

    # Create an instance of DataParser
    parser = DataParser(file_path)

    # Parse the file and obtain the formatted DataFrame
    formatted_data = parser.parse_file()

Author: Stian Knudsen
Date: 2024-01-05
"""

import pandas as pd
from pathlib import Path


class DataParser:
    """
    A class to parse input files containing sensor data and format it into a Pandas DataFrame.

    Attributes:
        filename (Path): The path to the file to be parsed.

    Methods:
        parse_file(): Parse the input file and format the data into a Pandas DataFrame.
    """

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
        try:
            with open(self.filename, "r") as file:
                # Read the file and store its contents in a list.
                lines = file.readlines()
        except IOError as err:
            raise IOError(f"Error reading the file {self.filename}: {err}")

        # Remove newlines.
        while "\n" in lines:
            lines.remove("\n")

        # Remove header and footer.
        lines = lines[3:-4]

        # Remove all instances of ":av" and fix missing headers.
        lines = [line.replace(":av", "") for line in lines]
        lines[1] = "Time of sensors\t" + lines[1]
        lines[2] = "localtime\t" + lines[2]

        # Merge lines that have AM/PM separated, Scada generation bug.
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

        # Set the index to the datetime
        df.set_index(column_tuples[0], inplace=True)

        return df
