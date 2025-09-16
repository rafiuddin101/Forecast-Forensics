# Forecast Forensics

A Python toolkit for analyzing and visualizing forecast reliability, decomposition, and calibration.

## Overview

Forecast Forensics provides tools to evaluate forecasting models through:
- Reliability analysis
- Score decomposition
- R* calculation
- Visualization of results

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Forecast.git
cd Forecast

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The package provides a command-line interface for easy analysis:

#### Reliability Analysis

```bash
python -m Forecast.models.cli reliability --csv data/synthetic_hetero.csv --y y --pred mean_pred --functional mean --n_bins 20 --plot
```

Options:
- `--csv`: Path to CSV file with data
- `--y`: Column name for actual values
- `--pred`: Column name for predictions
- `--functional`: Type of functional (mean, quantile, proba)
- `--alpha`: Alpha value for quantile (required for quantile functional)
- `--n_bins`: Number of bins for analysis
- `--plot`: Generate visualization
- `--output`: Save plot to file (e.g., reliability.png)

#### Decomposition Analysis

```bash
python -m Forecast.models.cli decompose --csv data/synthetic_hetero.csv --y y --pred mean_pred --functional mean --n_bins 20 --plot
```

Options:
- Same as reliability, plus decomposition-specific options

## Visualization Examples

The toolkit now includes visualization capabilities:

1. **Reliability Plots**: Show the relationship between predicted values and actual empirical means
2. **Decomposition Plots**: Display score components (UNC, DSC, MCB) and R* value

## Project Structure

- `reliability.py`: Reliability analysis functions
- `decomposition.py`: Score decomposition functions
- `rstar.py`: R* calculation
- `plots/`: Visualization modules
- `models/cli.py`: Command-line interface
- `data/`: Sample datasets

## License

[Your License]