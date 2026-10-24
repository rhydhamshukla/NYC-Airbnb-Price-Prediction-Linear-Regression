# NYC Airbnb Price Prediction using Linear Regression

**Student:** Rhydham Shukla  
**Course:** Machine Learning Program, Unlox Academy  
**Project:** Minor Project 3  

## Project Overview
In this project, I build an end to end Linear Regression pipeline to predict nightly Airbnb prices in New York City using the official Kaggle NYC Airbnb Open Data (48,895 raw rows).

The goal is to understand how basic listing attributes such as borough location, room type, minimum nights, and review counts drive rental prices, while addressing real world challenges like missing values, extreme outliers, and categorical feature encoding without falling into the curse of dimensionality.

## Key Model Evaluation Results
* R Squared (R2) Score: 0.4868 (48.7% variance explained)
* Mean Absolute Error (MAE): $38.32
* Root Mean Squared Error (RMSE): $52.54
* Training Sample Size: 36,598 listings (80% split)
* Testing Sample Size: 9,150 listings (20% split)

## Plain English Interpretation
An R2 score of 0.487 means that our linear regression model accounts for approximately 48.7% of the variation in nightly listing prices using basic listing details. The Mean Absolute Error (MAE) shows that on average, our predictions are within $38 of the true listing price set by the host.

## Engineering Decisions
1. **Handling Missing Values:**
   * `last_review` was missing in 10,052 rows (over 20%). Because it is a date string and review activity is already represented by review counts, I removed it.
   * `reviews_per_month` was missing in the exact same 10,052 rows where `number_of_reviews` was 0. I imputed these missing values with 0.0 because no reviews were recorded.

2. **Outlier Filtering (95th Percentile Rule):**
   * Rare luxury penthouses listed up to $10,000 per night and long term leases of 365+ nights distort linear regression.
   * Applying 95th percentile caps at $355.00 per night and 30 minimum nights retained 45,748 listings (93.56% of total data), focusing the model on typical consumer stays.

3. **Avoiding High Cardinality and Multicollinearity:**
   * Encoding all 221 specific neighborhoods would create over 220 sparse columns, causing the curse of dimensionality.
   * I retained `neighbourhood_group` (5 NYC boroughs) and `room_type`, using `drop_first=True` during one hot encoding to prevent the dummy variable trap.

4. **Performance by Price Segment:**
   * The model predicts budget and mid tier listings ($40 to $150) much more accurately than luxury listings ($200 to $355).
   * Over 75% of New York Airbnb listings are budget and mid tier apartments in Brooklyn, Queens, and Upper Manhattan, so the linear regression weights focus on this central density.
   * Luxury rentals depend heavily on uncaptured features such as skyline views, interior styling, premium furnishings, and seasonal demand.

## Project Structure
* `NYC_Airbnb_Price_Prediction.ipynb`: Fully executed Jupyter notebook with data auditing, visual plots, and metrics.
* `Rhydham_Shukla_Minor_Project_3_Report.pdf`: One page summary report for mentors and business stakeholders.
* `requirements.txt`: Python package requirements.

## How to Run
1. Clone this repository:
   git clone https://github.com/rhydhamshukla/NYC-Airbnb-Price-Prediction-Linear-Regression.git
2. Install dependencies:
   pip install -r requirements.txt
3. Place `AB_NYC_2019.csv` in the project root directory.
4. Launch the notebook:
   jupyter notebook NYC_Airbnb_Price_Prediction.ipynb
