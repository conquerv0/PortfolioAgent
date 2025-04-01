# Prediction Evaluation Framework

This module provides a framework for evaluating agent predictions across different asset classes.

## Overview

The evaluation system is designed with a base `Evaluator` class that provides common functionality, and specialized subclasses for each asset class:

- `FixedIncomeEvaluator`: For evaluating treasury/fixed income predictions
- `ForexEvaluator`: For evaluating foreign exchange (forex) predictions

## Usage

### Running Evaluations

To run evaluations, use the `run_evaluations.py` script:

```bash
# Run both evaluations with default settings
python -m src.backtest.run_evaluations

# Run only fixed income evaluation
python -m src.backtest.run_evaluations --eval-type fi

# Run only forex evaluation
python -m src.backtest.run_evaluations --eval-type fx

# Specify custom file paths
python -m src.backtest.run_evaluations --fi-pred custom_fi_predictions.csv --fi-data custom_fi_data.csv
```

### Command Line Arguments

- `--eval-type`: Type of evaluation to run (`fi`, `fx`, or `all`)
- `--fi-pred`: Path to fixed income predictions file
- `--fi-data`: Path to fixed income actual data file
- `--fx-pred`: Path to forex predictions file
- `--fx-data`: Path to forex actual data file
- `--output-dir`: Base directory for saving evaluation results

## Input Data Format

### Prediction Files

Prediction files should be CSV files with the following columns:

- `date`: Date of the prediction
- `predicted_return`: The predicted return value
- `confidence`: Confidence score for the prediction
- Asset identifier column:
  - For Fixed Income: `instrument` (e.g., "Short-Term Treasury", "1-3 Year Treasury")
  - For Forex: `etf` (e.g., "FXE", "FXB") and `currency_pair` (e.g., "EUR/USD", "GBP/USD")

### Actual Data Files

Actual data files should contain price data for each instrument over time, with:
- A date column
- Price columns named after the asset identifiers (e.g., "SHV", "FXE")

## Output

Evaluations produce:

1. Performance metrics:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R-squared
   - Correlation
   - Directional Accuracy

2. Trading strategy performance:
   - Directional strategy
   - Confidence-weighted strategy
   - Size-proportional strategy

3. Visualizations:
   - Predicted vs actual returns
   - Cumulative returns by instrument
   - Confidence vs accuracy
   - Strategy performance comparison

## Extending the Framework

To add support for a new asset class:

1. Create a new subclass of `Evaluator`
2. Implement the required abstract methods:
   - `prepare_evaluation_data()`
   - `calculate_trading_performance()`
   - `create_visualizations()`
3. Update the `run_evaluations.py` script to include the new evaluator 