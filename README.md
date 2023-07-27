# US IMMIGRATION PROJECT

![AdobeStock-385843648-US+Immigration copy](https://github.com/ehsanshahrabi/project4/assets/122590244/c66850e3-7859-40e4-852c-f0e9f180dff8)

## Team Members
Ehsan Shahrabi, Mirian Ruanova, Yi Pan & Amna Hussain

## Project Overview
We’re focusing on US Immigration over 15 years from 8 countries specifically: China, Dominican Republic, United Kingdom, Mexico, Iran,
India, Pakistan and Philippines. From those countries, we want to determine how many people are migrating to the US and which state they are settling to. To do this, we'll create a Machine Learning model that is able to predict the number of permanent residents for the next couple years using data from the Department of Homeland Security, US Census Bureau and the World Bank Group. 

## Immigration Statistics:
We wanted to explore the concept of immigration and its significance before diving into our data. The US Census Bureau reveals that almost 20% of the world’s migrants live in the United States and there are 45.3 million immigrants in the United States as of 2021. That’s 13.6% of the total U.S. population, just below the 13.7% high in 2019. People from Mexico, China, and India are the largest US immigrant groups. Immigrants from Mexico have been the most numerous since 1980, whereas the influx of immigrants from India and China have only grown since 2013. In general, there are more female than male immigrants coming in to the US with about 51% of all U.S. immigrants being female in 2021. 18 million out of 69.7 million U.S. children under the age of 18 lived with atleast one immigrant parent in 2021. Overall, 1,031,765 people immigrated to the US and became permanent residents in 2019. 

## Data Sources, Data Delivery, and ETL Process:

The dataset used in this notebook, Ehsan's Portion/immigration_Data_and_data_delivery_ETL.ipynb, encompasses immigration data from 2005 to 2019. This data includes records of region and country of birth of immigrants, and the total number of permanent residents for each year.

The data is delivered in Excel file format with each file corresponding to a single year. The data files are stored in a directory named Resources (in Ehsan's Portion)

#### The ETL (Extract, Transform, Load) process for the data consists of the following steps:

Extraction: The data is extracted from multiple Excel files, each corresponding to a specific year.

Transformation: The data is cleaned, transformed, and standardized to create a master dataframe. The transformation steps include:

Extracting the year from the file name and adding it as a column in the DataFrame

Replacing missing values or data withheld for privacy reasons with zero

Removing unnecessary columns and rows

Renaming the 'Total' column to 'Total Permanent Residents'

Removing duplicate entries

Adding a new column 'Percentage' that calculates the percentage of immigrants from each country with respect to the total number of immigrants for a given year.

Rounding the 'Percentage' values to two decimal places

Identifying the top 10 countries by percentage of total immigration for each year

Load: The final cleaned data is saved to a new CSV file. Ehsan's Portion/Resources/immigration_selected_2005_2019.csv

Additionally, it is also creat a SQLite database to meet project requirements.Ehsan's Portion/Resources/immigration_selected_2005_2019_sqlite.sqlite

## Predictions with Machine Learning steps, Analytics and Visualization: 
We start perediction and find final ML model with following Jupiter notbook files:

1- In Mirian's folder : Prophel Model (Mirian's folder/Prophet_2019_metrics.ipynb)

2- In Yi's Portion : Neural network model  (Yi's Portion/neural.ipynb)

3- in Ehsan's Portion : These notebooks include different machine learning models such as ARIMA, SARIMAX, RandomForestRegressor, and Linear Regression. ( Ehsan's Portion/ML-ARIMA.ipynb, Ehsan's Portion/ML-RandomForestRegressor.ipynb, Ehsan's Portion/ML-SARIMAX.ipynb, Ehsan's Portion/ML-LinearRegression.ipynb)

4- Final Machine Learning model.

5- Analytics and Visualization.

### 1- Prophet model:

#### Data Sources

1. **Migration Data**: The main dataset used in this analysis is stored in the CSV file "immigration_data_2012_2021.csv". This dataset contains information on immigration to the United States from various countries for the years 2012 to 2021.

2. **Country Regressors**: The dataset "countries_metadata.csv" contains various indicators and attributes for different countries. We use a subset of this data to create regressors for our migration analysis.

3. **US State Regressors**: The dataset "variable 2011-2021.csv" provides variables and attributes for different US states. We use this data to create additional regressors for predicting migration trends.

#### Data Cleaning

The code first reads and preprocesses the country regressor dataset and the US state regressor dataset to create cleaned and filtered DataFrames with relevant columns.

#### Database Creation

To facilitate data retrieval and storage, the code creates an SQLite database named "use_migration.db" and stores the preprocessed DataFrames in this database. This enables easy access to data for future analyses without the need to read CSV files repeatedly.

#### Migration Analysis

The main part of the analysis involves predicting future migration trends using machine learning and time series forecasting techniques. We use a Random Forest Regressor to identify the most important features, and then utilize the Prophet library for time series forecasting.

For each combination of country and US state, the code performs the following steps:

1. Merging the migration data with country and state regressors.
2. Preparing and cleaning the data for use with Prophet.
3. Creating a Random Forest Regressor to identify important features for forecasting.
4. Setting up a Prophet model with additional regressors.
5. Fitting the Prophet model to the data.
6. Predicting migration values for the year 2019 and 2023 using the trained model.
7. Calculating metrics such as Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE), Median Absolute Error (MEDAE), and Coefficient of Variation of RMSE (CVRMSE) for the predictions.

#### Random Forest Regressor (Data Model Optimization)

The analysis uses a *Random Forest Regressor* to identify the *three most important features* for predicting migration trends. These important features are then included as regressors in the Prophet model to enhance the prediction accuracy.

[Feature importance](https://github.com/ehsanshahrabi/project4/blob/mirian/print_feature_importance.JPG)

#### Setting up the Prophet Model

Prophet is set up with the three most important regressors from the Random Forest Regressor, along with a `Holiday` regressor to model the COVID-19 effect. The "Holiday" regressor is not included in the forecast to avoid future influence.

#### Data Transformation

To improve model performance and *avoid training the model with negative values*, the code performs a logarithmic transformation only for the population data before training the Prophet model `prophet_df_pred['y'] = np.log1p(prophet_df_pred['y']`. After training, an exponential transformation is applied to obtain the population data for predictions `predicted_values = np.exp(forecast['yhat']`.

#### Model Evaluation

The Mean Absolute Percentage Error (MAPE) is used as the evaluation metric for the Prophet model's predictions. The MAPE measures the accuracy of the model in predicting the future migration numbers. We have created  a *heatmap* to analyze and compare the overall MAPE results.

![Hetmap2019](https://github.com/ehsanshahrabi/project4/assets/124327258/49d2eb37-e37f-4c3f-b71d-151129cf8490)

Migration Prediction for Multiple Countries and States:

The analysis is performed for 7 countries and 50 US states, resulting in a total of 350 predictions. The results are stored in a DataFrame named "results_df," which includes various metrics for each country and US state combination. 

### 2- Neural network model  :

Data:

The data used in these notebooks is stored in an SQLite database (Resources/immigration_selected_2005_2019_sqlite.sqlite), which includes immigration information from various countries to multiple states in the US between 2005 and 2019.

The dataset (sample.csv) contains the following columns: 'Year', 'Country Name', 'State', 'Fertility rate_country', 'GDP per capita_country', 'Gini index_country', 'Unemployment_country', 'Unemployment Rate', 'per capita personal Income', and 'Population Count'.

The neural network model is built using TensorFlow and Keras. The architecture of the model is as follows:

An input layer of 64 neurons with a ReLU activation function.

A hidden layer of 32 neurons with a Sigmoid activation function.

An output layer with a single neuron as the prediction of population count.

The model is trained for 100 epochs with a batch size of 32. The Adam optimizer and Mean Squared Error (MSE) loss function are used during the training of the model.

The input features are scaled using the StandardScaler from the sklearn.preprocessing module to ensure that all features have a similar scale.

The dataset is split into a training and a test set with 80% of the data used for training and 20% for testing. The split is stratified based on the 'Year' feature to ensure that the training and test sets have a similar distribution of data across the years.

The model's performance is evaluated using various error metrics, including Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Variation of Root Mean Squared Error (CVRMSE).

### 3- ARIMA, SARIMAX, RandomForestRegressor, and Linear Regression: 

#### ML-LinearRegression.ipynb: 

The main Dataset is immigration_selected_2005_2019_sqlite.sqlite

This notebook uses a Linear Regression model to predict the immigration data. It's noted for its simplicity, robustness, and reasonably accurate predictions.

Outer Loop: This loop iterates over a list of unique countries (countries). For each country, it subsets the dataset (df) to only include the rows where the 'Region and country of birth' is the current country. This resulting subset is stored in df_country, which is used in the inner loop.

Inner Loop: The inner loop iterates over a list of all states (states). For each state, it prepares the state-specific data by setting X to be the 'Year' column and y to be the column for the current state's immigration data in df_country. This data is then used to fit a linear regression model and make predictions for each state in each country.

Data Processing and Model Fitting:

Inside the inner loop, for each state-country pair, the following steps are performed:

Data Normalization: 

The 'Year' feature is normalized using MinMaxScaler to bring all values within the range [0, 1]. This process ensures the model does not bias toward features with larger numerical values.

Train-Test Split: The data is split into training and testing sets using a 80-20 split. This allows us to train our model on the majority of the data, and then test it on unseen data to evaluate its performance.

Model Fitting and Predictions: 

A Linear Regression model is fit on the training data, then used to predict on the test data (y_pred). The true and predicted values are stored in all_true and all_preds lists, respectively, for further calculation of evaluation metrics.

Metrics Calculation: The Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE), Mean Squared Error (MSE), and Coefficient of Variation of the RMSE (CVRMSE) are calculated for each state-country pair and appended to the df_scores DataFrame.

Future Predictions: 

The model is used to make immigration predictions for the next years (2020 to 2025), and the predictions are stored in the df_predictions DataFrame.

The loop structure allows the script to efficiently process a large amount of data and apply the same procedures to each country-state pair. Finally, the df_scores and df_predictions DataFrames are saved to .csv files, providing the model's performance metrics and immigration predictions for future use.

#### ML-ARIMA.ipynb: 
This script employs the ARIMA model, a widely-used forecasting method that leverages information from past values for future predictions. The loop structure for data preprocessing is akin to ML-LinearRegression.ipynb, with minor adjustments to accommodate the ARIMA model's requirements.

#### ML-RandomForestRegressor.ipynb: 
This notebook utilizes the Random Forest Regressor, a versatile ensemble learning method known for its high predictive power. The data processing loop structure mirrors the one used in ML-LinearRegression.ipynb, with changes in model fitting and predictions to accommodate the Random Forest Regressor.

#### ML-SARIMAX.ipynb: 
This script leverages the SARIMAX model, which extends the ARIMA model by incorporating seasonal components and exogenous regressors. The loop structure used for data preprocessing is similar to the other scripts but is adjusted to meet the SARIMAX model's needs.

### 4- Final Machine Learning model:

#### Best Performing Model

Upon rigorous testing and validation, the Linear Regression model (ML-LinearRegression.ipynb) yielded the best performance overall. Factors such as model robustness, simplicity, and its capability of generating accurate predictions contributed to its superior performance compared to the other models.

However, the choice of model may vary depending on the specifics of the problem at hand and the nature of the data. It's recommended to consider other models and approaches and choose the one that best aligns with your needs.

For a detailed walkthrough of the ML-LinearRegression.ipynb script, please refer to the earlier section in this README file. For the remaining notebooks, you can refer to the inline comments within each script for an in-depth understanding of the individual steps.

Performance Evaluation for Predictions
Evaluating the performance of the model is a critical step in the machine learning pipeline. It enables us to understand the efficacy of our model in making reliable and accurate predictions.

In these notebooks, we initially use a dataset spanning from 2005 to 2019 to train our models and generate predictions. After this initial analysis, we further test our model's performance by using a slightly modified dataset.

In the modified dataset, we exclude the data for the year 2019 and retrain our model using data only up to 2018. We then use our trained model to generate predictions for the year 2019.

The true test of a model's performance lies in its ability to predict unseen data accurately. Therefore, we compare the predictions made by the model for 2019 with the actual data from 2019, which we had excluded earlier. This gives us a measure of how well our model is likely to perform on new, unseen data.

By comparing the actual data of 2019 with the predictions, we get a concrete idea of the model's performance and its ability to handle real-world, future data. This also helps us to understand whether our model is underfitting or overfitting the training data and allows us to make necessary adjustments to improve the model's performance.

![Screenshot 2023-07-26 220636](https://github.com/ehsanshahrabi/project4/assets/124327258/8f25dfdc-dde3-4f72-9834-fd44565f5e16)

### 5- Analytics & Visualizations powered by Tabelou:

Predictions for immigrants from China in 2024 and 2025

![Screenshot 2023-07-26 221221](https://github.com/ehsanshahrabi/project4/assets/124327258/af398343-5a1f-4dec-b12a-2cf5748d5e72)

Predictions for immigrants from Dominican Republic in 2024 and 2025

![Screenshot 2023-07-26 221301](https://github.com/ehsanshahrabi/project4/assets/124327258/90a6c612-0774-4de0-ac71-4f27717fa0c6)

Predictions for immigrants from India in 2024 and 2025

![Screenshot 2023-07-26 221348](https://github.com/ehsanshahrabi/project4/assets/124327258/fcbf7d07-f5e9-4b88-82df-e94a3602fa1f)

Predictions for immigrants from Mexico in 2024 and 2025

![Screenshot 2023-07-26 221413](https://github.com/ehsanshahrabi/project4/assets/124327258/849b87c2-cdfb-4381-a9b2-ba935fed0ac9)

Predictions for immigrants from Pakistan in 2024 and 2025

![Screenshot 2023-07-26 221519](https://github.com/ehsanshahrabi/project4/assets/124327258/e3c3ae88-8e07-48cb-83b0-522bf6a29e0b)

Predictions for immigrants from Philippines in 2024 and 2025

![Screenshot 2023-07-26 221655](https://github.com/ehsanshahrabi/project4/assets/124327258/ced2878b-a02a-43ca-846a-bed8aacc3dfa)

Predictions for immigrants from United Kingdom in 2024 and 2025

![Screenshot 2023-07-26 221705](https://github.com/ehsanshahrabi/project4/assets/124327258/e186b9c1-c472-4f27-83e1-2ea16291a130)


#### Analytics:
From our 8 countries, the top 5 states vary slightly with California, New York, New Jersey and Texas almost always being present. 

For the same 4 states above, immigrants make up more than 20% of their state population.

Based on our predicted data, we can see that there will be an increase of Chinese and Indian immigrants from their previous years – majority of both settling in California.

We’re also predicting that the number of Filipino, Mexican and British immigrants will stay roughly the same as their previous years.

Pakistani immigrants, in contrast, are the only group we predict will decrease in its number. 

### Disclaimer
This project is educational and should not be used to predict actual immigration patterns. Models are trained on historical data (2005-2019) and do not account for future unknowns such as policy changes, economic shifts, or geopolitical events. The predictions should not be used for actual decision-making or policy formation. Treat this project as a learning tool, not as a definitive predictor of future immigration trends.
 








