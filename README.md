# Stock Price Forecasting: Comparing Classical Statistical Models and Neural Networks
## Overview
This repository presents a comparative study of classical statistical models, such as a hybrid ARIMAX-EGARCH model, and neural network approaches for forecasting stock prices, particularly under conditions of data non-linearity and volatility. By identifying critical variables and exploring advanced learning algorithms, the research evaluates and highlights the performance differences between these two approaches.

## Findings
The study demonstrates that while neural networks like NARX are powerful tools for forecasting non-linear and volatile datasets, classical models such as ARIMAX-EGARCH offer superior and more robust performance for forecasting daily closing stock prices when proper preprocessing is applied.

## Repository Structure
### MATLAB Scripts:

Implements the NARX model for forecasting with flexible configurations for input delays, feedback delays, and hidden layer sizes.
Includes code for evaluating performance metrics such as RMSE, RMSLE, MAE, and MAPE.
### R Scripts:

Implements the ARIMAX-EGARCH model for forecasting, including diagnostic tests.
Includes code for evaluating performance metrics such as RMSE, RMSLE, MAE, and MAPE.
## Data

Contains daily data from Dr. Abidi Pharmaceutical Company, a member of the Tehran Stock Exchange (stock symbol: Dabid).
The dataset spans 7 years, collected from November 27, 2011, to March 20, 2019.
Preprocessing includes converting character columns to numeric values and applying logarithmic transformations.
## Performance Evaluation
The models are compared using the following metrics:

Root Mean Square Error (RMSE): Measures the average magnitude of prediction errors.
Root Mean Square Logarithmic Error (RMSLE): Evaluates logarithmic differences between actual and predicted values.
Mean Absolute Error (MAE): Measures the average of absolute prediction errors.
Mean Absolute Percentage Error (MAPE): Evaluates the percentage error relative to actual values.
## Future Work
Integrating hybrid models that combine statistical and machine learning approaches.
Exploring other neural network architectures like LSTM or GRU for time series forecasting.
## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for improvements or additional features.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
