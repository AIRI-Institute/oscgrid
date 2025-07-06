import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, Any, List, Optional, Tuple

# Utility function from previous code
def _sliding_window_fft(signal: np.ndarray, fft_window_size: int, num_harmonics: int) -> np.ndarray:
    n_points = len(signal)
    fft_results = np.full((n_points, num_harmonics), np.nan + 1j * np.nan, dtype=complex)
    if n_points < fft_window_size:
        return fft_results
    for i in range(n_points - fft_window_size + 1):
        window_data = signal[i : i + fft_window_size]
        fft_coeffs = np.fft.fft(window_data) / fft_window_size
        if num_harmonics >= 1 and 1 < len(fft_coeffs):
            fft_results[i, 0] = fft_coeffs[1] * 2
    return fft_results

class OscillogramEventSegmenter:
    """
    Class for finding and extracting "events" in oscillograms, represented as a DataFrame.
    Supports two modes: for pre-normalized and for "raw" signals.
    """
    def __init__(self,
                 fft_window_size: int,
                 config: Dict[str, Any],
                 signals_are_normalized: bool = False
                ):
        """
        Initializes the segmenter.

        Args:
            fft_window_size (int): FFT window size (number of points per period).
            config (Dict[str, Any]): Dictionary with configuration parameters.
            signals_are_normalized (bool): Flag indicating whether input signals are normalized.
        """
        self.fft_window_size = fft_window_size
        self.config = config
        self.signals_are_normalized = signals_are_normalized

        # General parameters
        self.detection_window_periods = self.config.get('detection_window_periods', 5)
        self.padding_periods = self.config.get('padding_periods', 10)
        self.current_patterns = [p.lower() for p in self.config.get('current_patterns', [])]
        self.voltage_patterns = [p.lower() for p in self.config.get('voltage_patterns', [])]

        # Mode-dependent parameters
        if self.signals_are_normalized:
            self.thresholds_current = self.config['thresholds_current_normalized']
            self.thresholds_voltage = self.config['thresholds_voltage_normalized']
        else:
            self.raw_analysis_config = self.config['raw_signal_analysis']
            self.thresholds_current = self.raw_analysis_config['thresholds_raw_current_relative']
            self.thresholds_voltage = self.raw_analysis_config['thresholds_raw_voltage_relative']

    def _get_target_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Finds columns for analysis in the DataFrame based on specified patterns."""
        target_cols = {'current': [], 'voltage': []}
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        for col_lower, orig_col_name in df_cols_lower.items():
            # Current patterns
            if any(p in col_lower for p in self.current_patterns):
                target_cols['current'].append(orig_col_name)
            # Voltage patterns
            elif any(p in col_lower for p in self.voltage_patterns):
                target_cols['voltage'].append(orig_col_name)
        
        return target_cols

    def _calculate_h1_amplitude_series(self, signal_values: np.ndarray) -> np.ndarray:
        """Calculates the time series of the first harmonic's amplitude."""
        if len(signal_values) < self.fft_window_size:
            return np.array([])
            
        fft_complex = _sliding_window_fft(signal_values, self.fft_window_size, num_harmonics=1)
        h1_complex_series = fft_complex[:, 0]
        h1_amplitude_series = np.abs(h1_complex_series)
        
        # Fill NaNs at the end for continuity
        valid_mask = ~np.isnan(h1_amplitude_series)
        if np.any(valid_mask):
            last_valid_value = h1_amplitude_series[valid_mask][-1]
            h1_amplitude_series[~valid_mask] = last_valid_value
        else:
            h1_amplitude_series.fill(0)
            
        return h1_amplitude_series

    def _get_clean_signal_h1_amplitude(self, signal_values: np.ndarray, channel_type: str) -> Optional[float]:
        """Checks the "cleanliness" of the initial signal segment and returns the base H1 amplitude."""
        cfg = self.raw_analysis_config
        periods_to_check = cfg['initial_window_check_periods']
        points_to_check = periods_to_check * self.fft_window_size

        if len(signal_values) < points_to_check:
            return None
        
        window_data = signal_values[:points_to_check]
        fft_coeffs = np.fft.fft(window_data) / len(window_data)
        
        idx_h1 = periods_to_check
        if idx_h1 >= len(fft_coeffs) // 2:
            return None

        m1_amplitude = np.abs(fft_coeffs[idx_h1]) * 2
        
        higher_harm_amplitudes = np.abs(fft_coeffs[idx_h1 + 1 : len(fft_coeffs) // 2 + 1]) * 2
        mx_amplitude = np.max(higher_harm_amplitudes) if len(higher_harm_amplitudes) > 0 else 0.0

        if mx_amplitude < 1e-9: # Avoid division by zero and consider the signal noisy
            return None
        
        ratio_h1_hx = m1_amplitude / mx_amplitude
        threshold = cfg[f'h1_vs_hx_ratio_threshold_{"I" if channel_type == "current" else "U"}']

        if ratio_h1_hx > threshold:
            return m1_amplitude
        return None

    def _mark_interesting_points(self, h1_series: np.ndarray, thresholds: Dict[str, float], use_max_abs_check: bool) -> np.ndarray:
        """Creates a boolean mask of interesting points for a single channel."""
        num_points = len(h1_series)
        mask = np.zeros(num_points, dtype=bool)
        detection_window_size = self.detection_window_periods * self.fft_window_size

        if num_points < detection_window_size:
            return mask

        for i in range(num_points - detection_window_size + 1):
            window_h1 = h1_series[i : i + detection_window_size]
            
            delta = window_h1.max() - window_h1.min()
            std_dev = window_h1.std()
            
            is_active_by_variation = (delta > thresholds['delta']) or (std_dev > thresholds['std_dev'])

            if use_max_abs_check:
                max_abs = window_h1.max()
                is_active_by_max = max_abs > thresholds['max_abs']
                is_active = is_active_by_variation and is_active_by_max
            else:
                is_active = is_active_by_variation

            if is_active:
                mask[i : i + detection_window_size] = True
        
        return mask

    def _find_and_merge_events(self, full_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Finds, expands, and merges continuous blocks in the mask."""
        if not np.any(full_mask):
            return []

        d = np.diff(full_mask.astype(int))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if full_mask[0]: starts = np.insert(starts, 0, 0)
        if full_mask[-1]: ends = np.append(ends, len(full_mask))
        if len(starts) == 0: return []

        padding_points = self.padding_periods * self.fft_window_size
        padded_zones = [[max(0, s - padding_points), min(len(full_mask), e + padding_points)] for s, e in zip(starts, ends)]
        
        padded_zones.sort(key=lambda x: x[0])
        merged = [padded_zones[0]]
        for current in padded_zones[1:]:
            last = merged[-1]
            if current[0] < last[1]:
                last[1] = max(last[1], current[1])
            else:
                merged.append(current)
        
        return [tuple(zone) for zone in merged]

    def process_single_dataframe(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Processes a single DataFrame, finding and extracting events."""
        if df.empty:
            return []

        file_name = df['file_name'].iloc[0]
        target_cols = self._get_target_columns(df)
        if not target_cols['current'] and not target_cols['voltage']:
            return []

        full_mask = np.zeros(len(df), dtype=bool)

        for channel_type, columns in target_cols.items():
            thresholds = self.thresholds_current if channel_type == 'current' else self.thresholds_voltage
            for col in columns:
                signal = df[col].values.astype(float)
                h1_series = self._calculate_h1_amplitude_series(signal)
                if len(h1_series) == 0:
                    continue
                
                h1_to_analyze = h1_series
                
                if not self.signals_are_normalized:
                    base_amplitude = self._get_clean_signal_h1_amplitude(signal, channel_type)
                    if base_amplitude is None or base_amplitude < 1e-6:
                        continue # Skip "noisy" or zero channel
                    h1_to_analyze = h1_series / base_amplitude
                
                channel_mask = self._mark_interesting_points(
                    h1_to_analyze, 
                    thresholds, 
                    use_max_abs_check=self.signals_are_normalized
                )
                full_mask |= channel_mask

        event_zones = self._find_and_merge_events(full_mask)

        result_dfs = []
        for i, (start, end) in enumerate(event_zones):
            event_df = df.iloc[start:end].copy()
            event_df['file_name'] = f"{file_name}_event_{i+1}"
            result_dfs.append(event_df)

        return result_dfs

    def process_csv_file(self, input_csv_path: str, output_csv_path: str):
        """Reads a CSV, processes all files within it, and saves the result to a new CSV.
           Also saves log files with lists of all processed files and files with found events.
        """
        if not os.path.exists(input_csv_path):
            print(f"Error: Input file not found: {input_csv_path}")
            return
        
        try:
            df_full = pd.read_csv(input_csv_path, index_col=False)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return

        all_event_dfs = []
        processed_file_names = []
        event_found_file_names = []

        grouped = df_full.groupby('file_name')
        
        for file_name, group_df in tqdm(grouped, desc="Processing files in CSV"):
            # Log that this file was processed
            processed_file_names.append(file_name)
            
            group_df = group_df.sort_index() 
            event_dfs_for_file = self.process_single_dataframe(group_df)
            
            if event_dfs_for_file:
                # If events are found, add them to the main list and log the file name
                all_event_dfs.extend(event_dfs_for_file)
                event_found_file_names.append(file_name)
        
        # --- Saving the main result (segmented events) ---
        if not all_event_dfs:
            print("\nNo final events found to save.")
            # Create an empty file with headers to indicate processing completion
            pd.DataFrame(columns=df_full.columns).to_csv(output_csv_path, index=False)
        else:
            final_df = pd.concat(all_event_dfs, ignore_index=True)
            try:
                final_df.to_csv(output_csv_path, index=False)
                print(f"\nFound and saved {len(all_event_dfs)} events.")
                print(f"Result saved to: {output_csv_path}")
            except Exception as e:
                print(f"\nError saving final CSV file: {e}")

        # --- Saving log files ---
        output_dir = os.path.dirname(output_csv_path)
        base_name = os.path.splitext(os.path.basename(output_csv_path))[0]

        # 1. Log of all processed files
        processed_log_path = os.path.join(output_dir, f"{base_name}_processed_log.csv")
        try:
            pd.DataFrame({'processed_files': processed_file_names}).to_csv(processed_log_path, index=False)
            print(f"List of all processed files ({len(processed_file_names)}) saved to: {processed_log_path}")
        except Exception as e:
            print(f"Error saving processed files log: {e}")

        # 2. Log of files with found events
        events_found_log_path = os.path.join(output_dir, f"{base_name}_events_found_log.csv")
        try:
            pd.DataFrame({'event_files': event_found_file_names}).to_csv(events_found_log_path, index=False)
            print(f"List of files with events ({len(event_found_file_names)}) saved to: {events_found_log_path}")
        except Exception as e:
            print(f"Error saving files with events log: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    config = {
        'detection_window_periods': 10, # 10 + 5 = 15 periods before/after event (300 ms for 50Hz or 250 ms for 60Hz)
        'padding_periods': 5,
        'current_patterns': ['ia', 'ib', 'ic', 'in'],
        'voltage_patterns': ['ua', 'ub', 'uc', 'un', 'uab', 'ubc', 'uca'],

        # --- Mode 1: Thresholds for PRE-NORMALIZED signals ---
        'thresholds_current_normalized': {
            'delta': 0.1/20, 'std_dev': 0.05/20, 'max_abs': 0.005/20
        },
        'thresholds_voltage_normalized': {
            'delta': 0.05/3, 'std_dev': 0.025/3, 'max_abs': 0.05/3
        },

        # --- Mode 2: Parameters for "RAW" signal analysis ---
        'raw_signal_analysis': {
            'initial_window_check_periods': 2,
            'h1_vs_hx_ratio_threshold_U': 10, # H1 should be 10 times greater than higher harmonics
            'h1_vs_hx_ratio_threshold_I': 1.5,  # H1 should be 1.5 times greater than higher harmonics
            # Thresholds for RELATIVE changes (fractions of base amplitude)
            'thresholds_raw_current_relative': {
                'delta': 0.4, 'std_dev': 0.2
            },
            'thresholds_raw_voltage_relative': {
                'delta': 0.05, 'std_dev': 0.025
            }
        }
    }

    # --- RUN: NORMALIZED DATA ANALYSIS ---   
    print("\n--- RUN: NORMALIZED DATA ANALYSIS ---")
    ## !!IMPORTANT!! 
    # Requires setting the correct number of points per period
    FFT_WINDOW_SIZE = 12 # e.g., 600 samples / 50 Hz = 12 samples/period
    # Specify file paths
    input_filepath_normalized = "./data/input_normalized_signals.csv"
    output_filepath_segmented = "./data/output_segmented_events.csv" # Name of the final output file
    
    segmenter_norm = OscillogramEventSegmenter(
        fft_window_size=FFT_WINDOW_SIZE,
        config=config,
        signals_are_normalized=True # Explicitly set the mode
    )
    segmenter_norm.process_csv_file(input_filepath_normalized, output_filepath_segmented)