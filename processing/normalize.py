import os
import pandas as pd



class NormOsc:
    # TODO: Consider that we have multiple "__init__" methods
    def __init__(self, norm_coef_file_path='OscGrid_dataset/Description/norm_coef_v1.3.csv'):
        if os.path.exists(norm_coef_file_path):
            with open(norm_coef_file_path, "r") as file:
                self.norm_coef = pd.read_csv(file, encoding='utf-8', low_memory=False)
                
    # TODO: Consider unifying this implementation, currently it's a local solution
    # Temporarily copied from raw_to_csv
    def normalize_bus_signals(self, raw_df, file_name, yes_prase = "YES", is_print_error = False):
        """Normalization of analog signals for each section."""
        norm_row = self.norm_coef[self.norm_coef["name"] == file_name] # Find normalization row by filename
        if norm_row.empty or yes_prase not in str(norm_row["norm"].values[0]): # Check if row exists and normalization is allowed
            if is_print_error:
                print(f"Warning: {file_name} not found in norm.csv file or normalization not allowed.")
            return None

        for bus in range(1, 9):
            nominal_current_series = norm_row.get(f"{bus}Ip_base")
            if nominal_current_series is not None and not pd.isna(nominal_current_series.values[0]):
                nominal_current = 20 * float(nominal_current_series.values[0])
                for phase in ['A', 'B', 'C']: # Current normalization
                    current_col_name = f'I | Bus-{bus} | phase: {phase}'
                    if current_col_name in raw_df.columns:
                        raw_df[current_col_name] = raw_df[current_col_name] / nominal_current

            nominal_current_I0_series = norm_row.get(f"{bus}Iz_base")
            if nominal_current_I0_series is not None and not pd.isna(nominal_current_I0_series.values[0]):
                nominal_current_I0 = 5 * float(nominal_current_I0_series.values[0])
                for phase in ['N']: # Zero-sequence current normalization
                    current_I0_col_name = f'I | Bus-{bus} | phase: {phase}'
                    if current_I0_col_name in raw_df.columns:
                        raw_df[current_I0_col_name] = raw_df[current_I0_col_name] / nominal_current_I0

            nominal_voltage_bb_series = norm_row.get(f"{bus}Ub_base")
            if nominal_voltage_bb_series is not None and not pd.isna(nominal_voltage_bb_series.values[0]):
                nominal_voltage_bb = 3 * float(nominal_voltage_bb_series.values[0])
                for phase in ['A', 'B', 'C', 'AB', 'BC', 'CA', 'N']: # BusBar voltage normalization
                    voltage_bb_col_name = f'U | BusBar-{bus} | phase: {phase}'
                    if voltage_bb_col_name in raw_df.columns:
                        raw_df[voltage_bb_col_name] = raw_df[voltage_bb_col_name] / nominal_voltage_bb
                    voltage_cl_col_name = f'U{phase} BB'
                    if voltage_cl_col_name in raw_df.columns:
                        raw_df[voltage_cl_col_name] = raw_df[voltage_cl_col_name] / nominal_voltage_bb

            nominal_voltage_cl_series = norm_row.get(f"{bus}Uc_base")
            if nominal_voltage_cl_series is not None and not pd.isna(nominal_voltage_cl_series.values[0]):
                nominal_voltage_cl = 3 * float(nominal_voltage_cl_series.values[0])
                for phase in ['A', 'B', 'C', 'AB', 'BC', 'CA', 'N']: # CableLine voltage normalization
                    voltage_cl_col_name = f'U | CableLine-{bus} | phase: {phase}'
                    if voltage_cl_col_name in raw_df.columns:
                        raw_df[voltage_cl_col_name] = raw_df[voltage_cl_col_name] / nominal_voltage_cl
                    voltage_cl_col_name = f'U{phase} CL'
                    if voltage_cl_col_name in raw_df.columns:
                        raw_df[voltage_cl_col_name] = raw_df[voltage_cl_col_name] / nominal_voltage_cl

            # TODO: Add differential current
            
        return raw_df