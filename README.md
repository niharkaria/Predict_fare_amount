
# Predicting Ride Fare Amounts Using Regression Analysis

## Project Overview

This project focuses on building a **regression model** to predict the fare amount of rides based on several factors such as:
- **Distance** between pickup and dropoff points
- **Ride duration**
- **Traffic conditions**
- **Time of day**
- **Passenger count**

By using historical ride data, we aim to provide insights into the key factors affecting ride fares and build a robust predictive model.

## Problem Statement

The fare amount for rides depends on multiple variables. This project seeks to analyze these factors and develop a model that can accurately predict fare prices for future rides. Such models can be beneficial for ride-hailing services to dynamically price rides based on real-time conditions.

## Dataset

The dataset contains 200,000 historical ride records, with the following columns:
- `fare_amount`: The fare paid for the ride
- `pickup_datetime`: Timestamp of when the ride started
- `pickup_longitude` and `pickup_latitude`: Geographic coordinates of the pickup location
- `dropoff_longitude` and `dropoff_latitude`: Geographic coordinates of the dropoff location
- `passenger_count`: Number of passengers in the ride

## Project Workflow

1. **Data Loading and Exploration**  
   - Load the dataset and explore its structure using Python libraries such as Pandas, NumPy, and Seaborn.

2. **Data Cleaning and Preprocessing**  
   - Handle missing values, convert date columns, and engineer relevant features like ride distance using the Haversine formula for calculating distances between geographic points.

3. **Exploratory Data Analysis (EDA)**  
   - Visualize relationships between variables such as distance, fare, and time using Seaborn and Matplotlib.
   - Analyze trends and outliers in the dataset to guide feature engineering.

4. **Feature Engineering**  
   - Derive new features such as:
     - **Ride distance** from pickup and dropoff coordinates
     - **Hour of day** and **day of the week** from the pickup timestamp
   - Handle geographical data to enhance prediction accuracy.

5. **Model Building**  
   - Train multiple regression models to predict the fare amount:
     - **Linear Regression**
     - **Random Forest**
     - **XGBoost**
   - Evaluate each model's performance using key metrics.

6. **Model Evaluation**  
   - Metrics used for model performance evaluation:
     - **Mean Squared Error (MSE)**
     - **Mean Absolute Error (MAE)**
     - **R² Score**

## Results

- **Best Model**: The XGBoost model achieved the highest accuracy with the following performance metrics on the test set:
  - **MSE**: 29.94
  - **MAE**: 2.31
  - **R² Score**: 0.71

## Key Insights

- **Distance** is the most significant factor affecting fare amount.
- **Time of day** (e.g., peak hours vs. non-peak hours) has a noticeable effect on fare pricing.
- Additional factors like **passenger count** show limited correlation with fare.

## Conclusion

This project demonstrates the potential of using historical data and machine learning models to predict future ride fares with reasonable accuracy. The model provides valuable insights into the dynamics of ride fare pricing and could be further refined by incorporating additional external factors, such as weather or real-time traffic data.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fare-prediction.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Project_Nihar.ipynb
   ```
4. Run the notebook cells sequentially to execute the code and visualize results.

## Future Work

- Integrate external features like weather and traffic data to further improve the model.
- Experiment with more advanced models, including **neural networks** or **deep learning techniques**.
- Deploy the model using a web interface or API for real-time fare predictions.

## Requirements

- Python 3.x
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn, XGBoost
- Jupyter Notebook
