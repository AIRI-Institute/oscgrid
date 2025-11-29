import os
import re
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np

from processing.comtrade_processing import ComtradeProcessing
from common.utils import get_short_names_ml_signals, get_short_names_ml_analog_signals, get_ml_signals
from processing.normalize import NormOsc

class RawToCSV():
    """
    This class implemented to convert raw comtrade files to csv file.
    """
    def __init__(self, raw_path=None, csv_path=None, uses_buses=['1', '2', '12']):
        # Use provided raw_path or default to labeled dataset
        if raw_path is None:
            # Default to labeled dataset
            labeled_dirs = glob.glob(os.path.join('OscGrid_dataset', 'Labeled_raw_v*'))
            if not labeled_dirs:
                raise FileNotFoundError("No Labeled_raw_v* directory found in OscGrid_dataset")
            self.raw_path = labeled_dirs[0]
        else:
            self.raw_path = raw_path

        # Set CSV path
        if csv_path is None:
            # Extract version from raw_path using robust method
            match = re.search(r'_v(\d+\.\d+)', self.raw_path)
            version = match.group(1)
            self.csv_path = os.path.join('OscGrid_dataset', f'CSV_format_v{version}')
        else:
            self.csv_path = csv_path

        # Add a separator at the end of the path if there is none.
        if not self.raw_path.endswith(os.path.sep):
            self.raw_path += os.path.sep
        if not self.csv_path.endswith(os.path.sep):
            self.csv_path += os.path.sep

        # Checking the existence of a directory with raw data
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Path for raw files does not exist: {self.raw_path}")

        # Creating a directory for CSV if it does not exist
        os.makedirs(self.csv_path, exist_ok=True)

        # Initializing a component for reading Comtrade files
        self.comtradeProcessing = ComtradeProcessing()

        # Getting dictionaries of signal names from comtradeProcessing
        self.analog_names_dict = self.comtradeProcessing.get_bus_names(analog=True, discrete=False)
        self.discrete_names_dict = self.comtradeProcessing.get_bus_names(analog=False, discrete=True)
        self.all_names = self.comtradeProcessing.get_all_names()

        # Settings for data processing
        self.uses_buses = uses_buses  # 12 - intersectional, is not taken into account in any way
        self.unread_files = set()

        # Settings for analog and digital signals
        # TODO: by uses variables - add truncation of signals from arrays.
        self.uses_CT_B, self.uses_CT_zero = True, True
        self.uses_VT_ph, self.uses_VT_iph, self.uses_VT_zero = True, True, True
        self.use_VT_CL, self.use_VT_BB = True, True
        # TODO: think about more clarity and optimality, since use_PDR is needed only now, and requires checking all the discretionary
        self.use_PDR = True

        # Parameters for signal processing
        # TODO: Add variables for combining accident levels (ML signals)
        self.number_periods = 10  # TODO: The number of samples is being set now. Think about a time-to-date task, or something similar.

        # Initialization of ML signal lists
        self.ml_all, self.ml_opr_swch, self.ml_abnorm_evnt, self.ml_emerg_evnt = get_short_names_ml_signals()

    def create_csv(self, csv_name='labeled.csv', is_cut_out_area=True, raw_files=None, normalize=False):
        """
        Creates a CSV file from raw COMTRADE data files.

        Args:
            csv_name (str): Name of the output CSV file.
            is_cut_out_area (bool): If True, removes non-ML signal areas from the data.
            raw_files (list): Optional list of specific .cfg files to process.
                            If None, processes all .cfg files in raw_path.
            normalize (bool): If True, normalizes signals using norm_coef file
        
        Returns:
            pd.DataFrame: The generated dataset.
        """
        dataset_df = pd.DataFrame()
        
        if normalize:
            norm_osc = NormOsc()
        # Get list of files to process
        if raw_files is None:
            raw_files = sorted([file for file in os.listdir(self.raw_path)
                              if file.endswith('.cfg')])
        with tqdm(total=len(raw_files), desc="Convert Comtrade to CSV") as pbar:
            for file in raw_files:
                # TODO: it is not rational to use two variables, but the necessary global data.
                # it will be necessary to think about optimizing this issue.
                raw_date, raw_df = self.comtradeProcessing.read_comtrade(self.raw_path + file)

                if normalize:
                    raw_df = norm_osc.normalize_bus_signals(raw_df, file[:-4])
                    if raw_df is None:  # Normalization failed
                        print(f"Warning: Normalization skipped for {file}")
                
                self.check_columns(raw_df, file)
                if not raw_df.empty:
                    raw_df = raw_df.reset_index()
                    buses_df = self._split_buses(raw_df, file)

                    frequency = raw_date.cfg.frequency
                    samples_rate = raw_date.cfg.sample_rates[0][0]
                    number_samples = int(samples_rate / frequency)  # TODO: It won't always be whole, but it's rare.
                    samples_before, samples_after = number_samples * self.number_periods, number_samples * self.number_periods
                    if is_cut_out_area:
                        buses_df = self.cut_out_area(buses_df, samples_before, samples_after)
                        dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=False)
                    else:
                        dataset_df = pd.concat([dataset_df, buses_df], axis=0, ignore_index=False)
                    pbar.update(1)

        # TODO: Think about organizing the processing itself differently and not adding a function.
        dataset_df = self.structure_columns(dataset_df)
        # if is_simple_csv:
        #     # We do not overwrite the original data, but only create another csv file
        #     self.get_simple_dataset(dataset_df.copy())

        dataset_df.to_csv(self.csv_path + csv_name, index=True, index_label='sample')
        return dataset_df

    def check_columns(self, raw_df, file):
        """check for unknown columns"""
        ml_signals = set()
        for i_bus in self.uses_buses:
            ml_signals.update(get_ml_signals(i_bus))

        all_names = self.all_names.union(ml_signals)
        columns = raw_df.columns
        for c in columns:
            if c not in all_names:
                raise NameError("Unknown column: " + c + " In file: " + file)


    def _create_one_df(self, file_path, file_name) -> pd.DataFrame:
        """
        The function of converting a single Comtrade file to pd.DataFrame().

        Args:
            file_path (str): The path to comtrade file.
            file_name (str): The name of comtrade file.

        Returns:
            pd.DataFrame: The DataFrame of the comtrade file.
        """
        dataset_df = pd.DataFrame()
        _, raw_df = self.comtradeProcessing.read_comtrade(file_path)
        self.check_columns(raw_df)
        if not raw_df.empty:
            raw_df = raw_df.reset_index()
            dataset_df = self._split_buses(raw_df, file_name)

        return dataset_df

    def _split_buses(self, raw_df, file_name):
        """Implemented for bus 1 and bus 2 only"""
        # Think about/consider handling cases where the names of the signals are the same. These are flaws in the standardization of names
        # This can happen often. For cases of "U|BusBar-1 | phase: A" and similar, this is often a division into
        # BusBar / CableLine.
        # For currents - sometimes it is divided into sections

        buses_df = pd.DataFrame()
        buses_cols = dict()
        raw_cols = set(raw_df.columns)
        for bus, cols in self.analog_names_dict.items():
            cols = raw_cols.intersection(cols)
            for i_bus in self.uses_buses:
                if len(bus[4:]) < 2 and (bus[-1] == i_bus or bus[-2] == i_bus):
                    ml_all = get_ml_signals(i_bus)
                    raw_ml = raw_cols.intersection(ml_all)
                    cols = cols.union(raw_ml)
            if cols:
                buses_cols[bus] = cols
        for bus, columns in buses_cols.items():
            bus_df = raw_df.loc[:, list(columns)]
            bus_df.insert(0, 'file_name', [file_name[:-4] + "_" + bus] * bus_df.shape[0])
            bus_df = self.rename_bus_columns(bus_df)
            buses_df = pd.concat([buses_df, bus_df], axis=0,
                                 ignore_index=False)
        return buses_df

    def rename_bus_columns(self, bus_df, is_use_ML=True, is_use_discrete=False):
        """
        This function renames columns in bus DataFrame.

        Args:
            buses_df (pandas.DataFrame): DataFrame with one bus.

        Returns:
            pandas.DataFrame: Dataframe with renamed columns.
        """
        bus_columns_to_rename = {}

        # Generate renaming for ML signals
        if is_use_ML:
            for i_bus in self.uses_buses:
                ml_signals = get_ml_signals(i_bus)
                for signal in ml_signals:
                    new_name = signal.replace(f'MLsignal_{i_bus}_', 'ML_')
                    bus_columns_to_rename[signal] = new_name

        # Generate renaming for analog signals
        for bus, names in self.analog_names_dict.items():
            for name in names:
                if 'I | Bus' in name:
                    phase = name.split(': ')[-1]
                    if (phase == 'B' and not self.uses_CT_B) or (phase == 'N' and not self.uses_CT_zero):
                        continue
                    bus_columns_to_rename[name] = f'I{phase}'

                elif self.use_VT_BB and 'U | BusBar' in name:
                    phase = name.split(': ')[-1]
                    if (((phase == 'A' or phase == 'B' or phase == 'C') and not self.uses_VT_ph) or
                            ((phase == 'AB' or phase == 'BC' or phase == 'CA') and not self.uses_VT_iph) or
                            (phase == 'N' and not self.uses_VT_zero)):
                        continue
                    bus_columns_to_rename[name] = f'U{phase} BB'

                elif self.use_VT_CL and 'U | CableLine' in name:
                    phase = name.split(': ')[-1]
                    if (((phase == 'A' or phase == 'B' or phase == 'C') and not self.uses_VT_ph) or
                            ((phase == 'AB' or phase == 'BC' or phase == 'CA') and not self.uses_VT_iph) or
                            (phase == 'N' and not self.uses_VT_zero)):
                        continue
                    bus_columns_to_rename[name] = f'U{phase} CL'

        # Generate renaming for discret signals
        if is_use_discrete:
            for bus, names in self.discrete_names_dict.items():
                for name in names:
                    if self.use_PDR and 'PDR | Bus' in name:
                        phase = name.split(': ')[-1]
                        bus_columns_to_rename[name] = f'PDR {phase}'

                        # TODO: signals I_raw, U_raw, I|dif-1, I | braking-1 are not taken into account

        bus_df.rename(columns=bus_columns_to_rename, inplace=True)
        return bus_df

    def cut_out_area(self, buses_df: pd.DataFrame, samples_before: int, samples_after: int) -> pd.DataFrame:
        """
        Cuts sections not containing ML signals, leaving samples before/after the signals.

        Args:
            buses_df: DataFrame to process.
            samples_before: Number of samples to keep before the first ML signal.
            samples_after: Number of samples to keep after the last ML signal.

        Returns:
            DataFrame with non-ML sections removed and file names annotated with event numbers.
        """
        ml_columns = [col for col in buses_df.columns if col in self.ml_all]
        dataset_dfs = []

        for file_name, bus_df in buses_df.groupby("file_name", sort=False):
            bus_df = bus_df.reset_index(drop=True)
            is_save = (bus_df[ml_columns] == 1).any(axis=1).to_numpy()

            # Handle case with no ML signals
            if not is_save.any():
                if len(bus_df) > samples_before + samples_after:
                    mid = len(bus_df) // 2
                    truncated = bus_df.iloc[mid - samples_before: mid + samples_after + 1]
                else:
                    truncated = bus_df
                dataset_dfs.append(truncated)
                continue

            # Extend regions around ML signals
            padded = np.concatenate([[False], is_save, [False]])
            diffs = np.diff(padded.astype(int))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0] - 1

            starts_adj = np.maximum(0, starts - samples_before)
            ends_adj = np.minimum(len(is_save) - 1, ends + samples_after)

            new_mask = np.zeros_like(is_save)
            for s, e in zip(starts_adj, ends_adj):
                new_mask[s:e+1] = True

            # Event numbering
            bus_df['is_save'] = new_mask
            event_starts = (~bus_df['is_save'].shift(fill_value=False)) & bus_df['is_save']
            
            bus_df['event'] = (
                event_starts.cumsum()
                .where(bus_df['is_save'])  # Only keep numbers where is_save is True
                .ffill()
                .fillna(0)  # Fill remaining NA with 0 (before first event)
                .astype(int)
            )

            bus_df.loc[bus_df['is_save'], 'file_name'] += ' _event N' + bus_df['event'].astype(str)
            truncated = bus_df[bus_df['is_save']].drop(columns=['is_save', 'event'])
            dataset_dfs.append(truncated)

        return pd.concat(dataset_dfs, ignore_index=False)

    def structure_columns(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function restructures the columns of the dataset for easier analysis.
        """
        # TODO: some names may be missing, so think about processing them.
        # Specify the desired order of columns
        desired_order = ["file_name"]
        desired_order.extend(get_short_names_ml_analog_signals())
        desired_order.extend(get_short_names_ml_signals()[0])
        desired_order.extend(["opr_swch", "abnorm_evnt", "emerg_evnt", "normal"])

        # Check if each column in the desired order exists in the DataFrame
        existing_columns = [col for col in desired_order if col in list(dataset_df.columns)]

        # Reorder the DataFrame columns based on the existing columns
        dataset_df = dataset_df[existing_columns]

        return dataset_df
