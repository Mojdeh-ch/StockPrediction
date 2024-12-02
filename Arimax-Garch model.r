# Load necessary libraries
library(readxl)       # For reading Excel files
library(xts)          # For working with time series data
library(forecast)     # For ARIMA models
library(tseries)      # For ADF/KPSS tests
library(ggplot2)      # For data visualization
library(gridExtra)    # For arranging multiple plots
library(rugarch)      # For GARCH models
library(lmtest)       # For diagnostic tests
library(tidyr)        # For data reshaping

# Load the data
data <- read_excel('data.xlsx', col_names = TRUE)

# Check structure of the data
str(data)

# Define a function to convert character columns to numeric
convert_char_to_num <- function(df) {
  df[] <- lapply(df, function(column) {
    if (is.character(column)) {
      as.numeric(column)
    } else {
      column
    }
  })
  return(df)
}

# Apply the conversion function
data <- convert_char_to_num(data)
str(data)

# Convert the date column to Date type
data$Date.AD <- as.Date(data$Date.AD)

# Convert the data to a time series object
ts_data <- xts(data[, -1], order.by = data$Date.AD)

# Split the data into training and testing sets
X <- ts_data[1:2086, 1:10]
Y <- ts_data[1:2086, 11]

# Plot the dependent variable
plot(Y, main = "Time Series of Y", xlab = "Time", ylab = "Y")

# Log-transform the data
x_train <- log(X[1:1773, ])
x_test <- log(X[1774:2086, ])
y_train <- log(Y[1:1773])
y_test <- log(Y[1774:2086])

# Function to perform ADF test and return p-value
adf_test_p_value <- function(series) {
  test_result <- adf.test(series)
  return(test_result$p.value)
}

# Perform ADF test on training data
adf_results <- sapply(x_train, adf_test_p_value)
print(adf_results)

# Check stationarity of dependent variable
plot(y_train, type = "l", main = "Log-transformed Y (Training Set)")
adf.test(y_train)
kpss.test(y_train)

# Plot ACF and PACF for the dependent variable
windows(10)
par(mfrow = c(2, 1))
acf(y_train, main = "ACF of Y")
pacf(y_train, main = "PACF of Y")

# Fit ARIMAX model
fit <- auto.arima(
  y_train, d = 1,
  xreg = cbind(x_train[, 1:10]),
  stationary = FALSE, seasonal = TRUE,
  stepwise = FALSE, approximation = FALSE
)
summary(fit)

# Extract residuals from ARIMAX model
residuals_arima <- residuals(fit)

# Create diagnostic plots for residuals
# 1. Time series plot of residuals
p1 <- ggplot(data.frame(Date = time(residuals_arima), Residuals = as.numeric(residuals_arima)), aes(x = Date, y = Residuals)) +
  geom_line(color = "blue") +
  labs(title = "Residuals Time Series", x = "Date", y = "Residuals") +
  theme_minimal()

# 2. Histogram of residuals
p2 <- ggplot(data.frame(Residuals = as.numeric(residuals_arima)), aes(x = Residuals)) +
  geom_histogram(fill = "blue", color = "black", bins = 30) +
  labs(title = "Histogram of Residuals", x = "Residuals", y = "Frequency") +
  theme_minimal()

# 3. ACF plot of residuals
acf_data <- acf(residuals_arima, plot = FALSE)
acf_df <- data.frame(lag = acf_data$lag, acf = acf_data$acf)
n <- length(residuals_arima)
conf_bound <- 1.96 / sqrt(n)
p3 <- ggplot(acf_df, aes(x = lag, y = acf)) +
  geom_bar(stat = "identity", fill = "blue", color = "black") +
  geom_hline(yintercept = c(conf_bound, -conf_bound), linetype = "dashed", color = "red") +
  labs(title = "ACF of Residuals", x = "Lag", y = "ACF") +
  theme_minimal()

# 4. Q-Q plot of residuals
p4 <- ggplot(data.frame(sample = residuals_arima), aes(sample = sample)) +
  stat_qq(color = "blue") +
  stat_qq_line(color = "red") +
  labs(title = "Q-Q Plot of Residuals", x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()

# Combine all diagnostic plots
grid.arrange(p1, p2, p3, p4, ncol = 2)

# Perform Shapiro-Wilk test for normality of residuals
shapiro.test(residuals_arima)

# Perform Ljung-Box test for autocorrelation
Box.test(residuals_arima, type = "Ljung-Box")

# Perform Breusch-Godfrey Test for serial correlation
bgtest(abs(residuals_arima) ~ fitted(fit))

# Perform ARCH Test for heteroscedasticity
ArchTest(residuals_arima)

# Forecast using ARIMAX model
forecast_result <- forecast(fit, xreg = x_test)

# Predict one step ahead with confidence intervals
y_hat <- forecast(fit, h = 1, level = c(80, 95), xreg = cbind(x_test[, 1:10]))

# Evaluate forecast accuracy
accuracy(y_hat, y_test)

# Plot the forecast vs actual values
Testing_Data <- ts(y_test,
                   start = start(forecast_result$mean),
                   frequency = frequency(forecast_result$mean))

autoplot(forecast_result) +
  autolayer(Testing_Data, series = "Actual") +
  xlab("Year") + ylab("Values") +
  ggtitle("Forecast vs Actual") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ACF and PACF plots of absolute residuals
windows(10)
par(mfrow = c(2, 1))
acf(abs(residuals_arima), main = "ACF of ARIMAX abs(Residuals)")
pacf(abs(residuals_arima), main = "PACF of ARIMAX abs(Residuals)")

# Define a function to fit and compare GARCH models
compare_garch_models <- function(data) {
  models <- list()
  aics <- matrix(NA, 5, 5)
  bics <- matrix(NA, 5, 5)
  
  for (p in 1:5) {
    for (q in 1:5) {
      spec <- ugarchspec(
        variance.model = list(model = "apARCH", garchOrder = c(p, q)),
        mean.model = list(armaOrder = c(0, 0)),
        distribution.model = "norm"
      )
      fit <- ugarchfit(spec = spec, data = data, solver = "hybrid")
      models[[paste(p, q, sep = ",")]] <- fit
      aics[p, q] <- infocriteria(fit)[1]  # AIC
      bics[p, q] <- infocriteria(fit)[2]  # BIC
    }
  }
  
  list(models = models, aics = aics, bics = bics)
}

# Fit GARCH models and compare AIC/BIC
results <- compare_garch_models(abs(residuals_arima))

# Print AIC and BIC matrices
print(results$aics)
print(results$bics)

# Identify the best GARCH model based on AIC
best_model_index <- which(results$aics == min(results$aics), arr.ind = TRUE)
best_p <- best_model_index[1]
best_q <- best_model_index[2]
cat("Best GARCH order based on AIC: (", best_p, ",", best_q, ")\n")

# Define and fit the chosen EGARCH model
spec <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(2, 3)),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "ged"
)
garch_fit <- ugarchfit(spec = spec, data = abs(residuals_arima), solver = "hybrid")
print(garch_fit)

# Summary and diagnostic plots of the GARCH model
summary(garch_fit)
plot(garch_fit, which = "all")
windows(10)
par(mfrow = c(2, 1))
acf(residuals(garch_fit, standardize = TRUE), main = "ACF of Standardized Residuals")
acf(residuals(garch_fit, standardize = TRUE)^2, main = "ACF of Squared Standardized Residuals")

# Plot conditional variance vs squared residuals
ug_var <- garch_fit@fit$var 
ug_res2 <- (garch_fit@fit$residuals)^2 
plot(ug_res2, type = "l", main = "Conditional Variance vs. Squared Residuals", 
     ylab = "Value", xlab = "Time")
lines(ug_var, col = "green")

# Perform additional diagnostic tests
standardized_residuals <- residuals(garch_fit, standardize = TRUE)
AutocorTest(standardized_residuals, lag = 10)
ArchTest(standardized_residuals, lag = 10)

# Forecast using the EGARCH model
garch_forecast <- ugarchforecast(garch_fit, n.ahead = nrow(y_test))

# Combine ARIMAX and EGARCH forecasts
predicted_volatility <- sigma(garch_forecast)
combined_forecast <- y_hat$mean + predicted_volatility

# Calculate accuracy metrics for models
fitted_arimax <- fitted(fit)
sigma_garch <- sigma(garch_fit)
fitted_combined <- fitted_arimax + sigma_garch

# Accuracy for ARIMAX and combined models on training and test sets
accuracy(exp(fitted(fit)), exp(y_train))  # ARIMAX train
accuracy(exp(y_hat$mean), exp(y_test))  # ARIMAX test
accuracy(exp(fitted_combined), exp(y_train))  # Combined train
accuracy(exp(combined_forecast), exp(y_test))  # Combined test

# RMSLE calculation function
rmsle <- function(actual, predicted) {
  log_diff <- log(predicted + 1) - log(actual + 1)
  sqrt(mean(log_diff^2))
}

# RMSLE for ARIMAX and combined models
rmsle_arimax_train <- rmsle(exp(as.numeric(y_train)), exp(as.numeric(fitted_arimax)))
rmsle_arimax_test <- rmsle(exp(as.numeric(y_test)), exp(as.numeric(y_hat$mean)))
rmsle_arimax_garch_train <- rmsle(exp(as.numeric(y_train)), exp(as.numeric(fitted_combined)))
rmsle_arimax_garch_test <- rmsle(exp(as.numeric(y_test)), exp(as.numeric(combined_forecast)))

cat("RMSLE for ARIMAX on Training Data: ", rmsle_arimax_train, "\n")
cat("RMSLE for ARIMAX on Test Data: ", rmsle_arimax_test, "\n")
cat("RMSLE for ARIMAX-GARCH on Training Data: ", rmsle_arimax_garch_train, "\n")
cat("RMSLE for ARIMAX-GARCH on Test Data: ", rmsle_arimax_garch_test, "\n")

# Load NARX model predictions from MATLAB
testPredictions_NARX <- read_excel('testPredictions_NARX.xlsx', col_names = TRUE)
NARX_prediction = t(as.matrix(testPredictions_NARX))

# Combine and compare results from ARIMAX, ARIMAX-GARCH, and NARX
df <- data.frame(
  Date = index(y_test),
  Actual = coredata(exp(y_test)),
  ARIMAX_forecasting = coredata(exp(y_hat$mean)),
  ARIMAX_GARCH_forecasting = coredata(exp(combined_forecast)),
  NARX_forecasting = coredata(NARX_prediction)
)
colnames(df) <- c("Date", "Actual", "ARIMAX_forecasting", "ARIMAX_GARCH_forecasting", "NARX_forecasting")

# Convert data to long format for visualization
df_long <- df %>% gather(key = "Type", value = "Value", -Date)

# Plot comparison of forecasts
ggplot(df_long, aes(x = Date, y = Value, color = Type)) +
  geom_line() +
  labs(title = "Comparison of Forecasted Values against Actual Values",
       x = "Date",
       y = "Values",
       color = "Legend") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_manual(values = c("purple", "blue", "green", "red"))