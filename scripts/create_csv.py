import sys
import os
import argparse
import glob
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.raw_to_csv import RawToCSV
from processing.normalize import NormOsc


def main():
    parser = argparse.ArgumentParser(description='Convert raw Comtrade data to CSV format.')
    parser.add_argument('--frequency', type=int, default=None, choices=[50, 60], help='Frequency of the network (50 or 60 Hz)')
    parser.add_argument('--sampling_rate', type=int, default=None, help='Sampling rate in Hz')
    parser.add_argument('--perturbations_only', action='store_true',
                      help='Only process files listed in perturbations CSV')
    parser.add_argument('--normalize', action='store_true',
                      help='Normalize signals using norm_coef file')
    args = parser.parse_args()

    # Handle unlabeled dataset case
    if args.frequency is not None and args.sampling_rate is not None:
        # Find unlabeled dataset version directory
        unlabeled_dirs = glob.glob(os.path.join('OscGrid_dataset', 'Unlabeled_raw_v*'))
        if not unlabeled_dirs:
            raise FileNotFoundError("No Unlabeled_raw_v* directory found in OscGrid_dataset")
        
        # Get version from first found directory (robust extraction)
        version = os.path.basename(unlabeled_dirs[0]).split('_v')[-1]
        
        # Construct input path
        input_path = os.path.join('OscGrid_dataset',
                                 f'Unlabeled_raw_v{version}',
                                 f'{args.frequency}hz_network',
                                 f'{args.sampling_rate}hz_sampling')
        
        # Generate output filename
        output_filename = f'unlabeled_{args.frequency}_{args.sampling_rate}.csv'
        
        # Create RawToCSV instance with constructed path
        raw_to_csv = RawToCSV(raw_path=input_path)
        
        if args.perturbations_only:
            # Find perturbations file
            perturbations_files = glob.glob(os.path.join('OscGrid_dataset', 'Description', 'with_perturbations_v*.csv'))
            if not perturbations_files:
                raise FileNotFoundError("No perturbations CSV file found in OscGrid_dataset/Description")
            
            # Read perturbations file
            perturbations_df = pd.read_csv(perturbations_files[0])
            perturbations_list = perturbations_df.iloc[:, 0].tolist()  # First column contains filenames
            
            # Filter raw files to only those in perturbations list
            raw_files = [f for f in os.listdir(input_path)
                        if 'cfg' in f and f[:-4] in perturbations_list]  # Remove .cfg extension for matching
            # Create CSV only with perturbations files
            raw_to_csv.create_csv(csv_name=output_filename,
                                  is_cut_out_area=False,
                                  raw_files=raw_files,
                                  normalize=args.normalize)
        else:
            # Create CSV with all files
            raw_to_csv.create_csv(csv_name=output_filename, is_cut_out_area=False, normalize=args.normalize)
    else:
        # Handle labeled dataset case (default behavior)
        raw_to_csv = RawToCSV()
        raw_to_csv.create_csv(normalize=args.normalize)


if __name__ == '__main__':
    main()