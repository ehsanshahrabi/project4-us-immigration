# project4

# Migration Data Analysis

This repository contains code for analyzing migration data from the years 2012 to 2021, along with various regressors for countries and US states. The analysis involves predicting future migration trends using machine learning and time series forecasting techniques.

## Data Sources

1. **Migration Data**: The main dataset used in this analysis is stored in the CSV file "immigration_data_2012_2021.csv". This dataset contains information on immigration to the United States from various countries for the years 2012 to 2021.

2. **Country Regressors**: The dataset "countries_metadata.csv" contains various indicators and attributes for different countries. We use a subset of this data to create regressors for our migration analysis.

3. **US State Regressors**: The dataset "variable 2011-2021.csv" provides variables and attributes for different US states. We use this data to create additional regressors for predicting migration trends.

## Data Cleaning

The code first reads and preprocesses the country regressor dataset and the US state regressor dataset to create cleaned and filtered DataFrames with relevant columns.

## Database Creation

To facilitate data retrieval and storage, the code creates an SQLite database named "use_migration.db" and stores the preprocessed DataFrames in this database. This enables easy access to data for future analyses without the need to read CSV files repeatedly.

## Migration Analysis

The main part of the analysis involves predicting future migration trends using machine learning and time series forecasting techniques. We use a Random Forest Regressor to identify the most important features, and then utilize the Prophet library for time series forecasting.

For each combination of country and US state, the code performs the following steps:
1. Merging the migration data with country and state regressors.
2. Preparing and cleaning the data for use with Prophet.
3. Creating a Random Forest Regressor to identify important features for forecasting.
4. Setting up a Prophet model with additional regressors.
5. Fitting the Prophet model to the data.
6. Predicting migration values for the year 2019 and 2023 using the trained model.
7. Calculating metrics such as Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE), Median Absolute Error (MEDAE), and Coefficient of Variation of RMSE (CVRMSE) for the predictions.

## Random Forest Regressor

The analysis uses a *Random Forest Regressor* to identify the *three most important features* for predicting migration trends. These important features are then included as regressors in the Prophet model to enhance the prediction accuracy.

[Feature importance](https://github.com/ehsanshahrabi/project4/blob/mirian/print_feature_importance.JPG)

## Setting up the Prophet Model

Prophet is set up with the three most important regressors from the Random Forest Regressor, along with a `Holiday` regressor to model the COVID-19 effect. The "Holiday" regressor is not included in the forecast to avoid future influence.

## Data Transformation

To improve model performance and *avoid training the model with negative values*, the code performs a logarithmic transformation only for the population data before training the Prophet model `prophet_df_pred['y'] = np.log1p(prophet_df_pred['y']`. After training, an exponential transformation is applied to obtain the population data for predictions `predicted_values = np.exp(forecast['yhat']`.

## Model Evaluation

The Mean Absolute Percentage Error (MAPE) is used as the evaluation metric for the Prophet model's predictions. The MAPE measures the accuracy of the model in predicting the future migration numbers. We have created  a *heatmap* to analyze and compare the overall MAPE results.

[Heatmap](https://github.com/ehsanshahrabi/project4/blob/mirian/Hetmap2019.JPG)

## Migration Prediction for Multiple Countries and States

The analysis is performed for 7 countries and 50 US states, resulting in a total of 350 predictions. The results are stored in a DataFrame named "results_df," which includes various metrics for each country and US state combination. 



## Dependencies

To run this code, you need the following Python libraries:
- numpy
- pandas
- datetime
- seaborn
- sqlite3
- matplotlib
- scikit-learn
- prophet


