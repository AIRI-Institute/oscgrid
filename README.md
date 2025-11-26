# A dataset of real-world oscillograms from electrical power grids (OscGrid)

This repository provides code for working with the [OscGrid](https://doi.org/10.6084/m9.figshare.28465427.v5) dataset.

## CSV creation from the raw data

Download the dataset into the repository folder and extract all the contained archives.

Create csv with labeled oscillograms:

```
python scripts/create_csv.py
```
To create a CSV with unlabeled oscillograms, use the following parameters:

<em>frequency</em> - Frequency of the network (50 or 60 Hz)

<em>sampling_rate</em> - Sampling rate in Hz

<em>perturbations_only</em> - Process files listed in perturbations CSV only

<em>normalize</em> - Normalize signals using norm_coef file

Example:

```
python scripts/create_csv.py --frequency 50 --sampling_rate 1800 --perturbations_only --normalize
```
