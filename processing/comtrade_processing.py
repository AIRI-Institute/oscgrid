import comtrade
import json
import os
import glob

class ComtradeProcessing:
    def __init__(self):
        """Initialize ReadComtrade class."""

        self.unread_files = set()

        # Getting the directory of the current file (comtrade_processing.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Creating full paths to JSON files
        analog_path = os.path.join(current_dir, "dict_analog_names.json")
        discrete_path = os.path.join(current_dir, "dict_discrete_names.json")
        #analog_path = glob.glob(os.path.join(current_dir, "dict_analog_names_v*.json"))[0]
        #discrete_path = glob.glob(os.path.join(current_dir, "dict_discrete_names_v*.json"))[0]

        try:
            # Loading data from JSON
            with open(analog_path, "r") as file:
                self.analog_names = json.load(file)
            with open(discrete_path, "r") as file:
                self.discrete_names = json.load(file)
        except FileNotFoundError as e:
            print(f"Error: Could not find JSON files: {e}")
            # Creating empty dictionaries in case of an error
            self.analog_names = {}
            self.discrete_names = {}

    def read_comtrade(self, file_name):
        """
        Load and read comtrade files contents.

        Args:
            file_name (str): The name of comtrade file.

        Returns:
            tuple:
            - raw_date (Comtrade): The raw comtrade
            - raw_df (pandas.DataFrame): DataFrame of raw comtrade file.
        """
        try:
            raw_date = comtrade.load(file_name)
            raw_df = raw_date.to_dataframe()
            return raw_date, raw_df
        except Exception as ex:
            self.unread_files.add((file_name, ex))
            return None, None

    def get_bus_names(self, analog=True, discrete=False):
        """
        This function makes a dict of analog and discrete names
        for each bus.

        Args:
            analog (bool): True - include analog names.
            discrete (bool): True - include discrete names.

        Returns:
            dict: dict of analog and discrete names for each bus.
        """
        bus_names = dict()
        if analog and hasattr(self, 'analog_names'):
            for bus in self.analog_names.keys():
                bus_names[bus] = set()
                for v in self.analog_names[bus].values():
                    bus_names[bus] = bus_names[bus].union(v)
        if discrete and hasattr(self, 'discrete_names'):
            for bus in self.discrete_names.keys():
                if bus not in bus_names:
                    bus_names[bus] = set()
                for v in self.discrete_names[bus].values():
                    bus_names[bus] = bus_names[bus].union(v)
        return bus_names

    def get_all_names(self):
        """
        This function makes a set of all analog and discrete names.

        Returns:
            set: set of all analog and discrete names.
        """
        all_names = set()
        buses_names = self.get_bus_names(discrete=True)
        for bus, names in buses_names.items():
            all_names = all_names.union(names)
        return all_names
