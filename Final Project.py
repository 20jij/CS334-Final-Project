import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import numpy as np

# Fetch stock data from Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Calculate additional features
def calculate_additional_features(symbol, data):
    additional_days = 500
    start_date = data.index[0] - pd.DateOffset(days=additional_days)
    end_date = data.index[-1]

    historical_data = yf.download(symbol, start=start_date, end=end_date)

    historical_data['MA_5'] = historical_data['Adj Close'].rolling(window=5).mean()
    historical_data['MA_21'] = historical_data['Adj Close'].rolling(window=21).mean()
    historical_data['MA_50'] = historical_data['Adj Close'].rolling(window=50).mean()
    historical_data['MA_100'] = historical_data['Adj Close'].rolling(window=100).mean()
    historical_data['MA_252'] = historical_data['Adj Close'].rolling(window=252).mean()

    MA_data = historical_data[historical_data.index >= '2000-01-03']
    data = pd.merge(data, MA_data[['MA_5', 'MA_21', 'MA_50', 'MA_100', 'MA_252']], left_index=True, right_index=True, how='left')
    
    return data

# Extract competitor/supplier stock prices
def fetch_competitor_stock_data(symbol, start_date, end_date):
    competitor_data = fetch_stock_data(symbol, start_date, end_date)
    competitor_data = competitor_data.rename(columns={
        'Open': f'{symbol}_Open',
        'High': f'{symbol}_High',
        'Low': f'{symbol}_Low',
        'Close': f'{symbol}_Close',
        'Adj Close': f'{symbol}_Adj_Close',
        'Volume': f'{symbol}_Volume'
    })
    return competitor_data

def feature_selection_pearson_corr(data, threshold):
    corr_matrix = data.corr()
    selected_features = corr_matrix['Adj Close'][corr_matrix['Adj Close'] > threshold].index
    data_filtered = data[selected_features]
    
    correlation_threshold = 0.9995
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
    data_filtered2 = data.drop(to_drop, axis=1)
    common_features = data_filtered.columns.intersection(data_filtered2.columns)
    data_filtered = data_filtered[common_features]

    data_filtered['Predict_Price'] = data['Predict_Price']

    return data_filtered


# Data preprocessing
def preprocess_data(data):
    filtered_data = data.loc[data.index >= '2011-01-01']
    data_filled = filtered_data.fillna(data.mean())
    data_filled = feature_selection_pearson_corr(data, 0.8)
    data_filled = data_filled.dropna(subset=['Predict_Price'])
    X = data_filled.drop('Predict_Price', axis=1)
    y = data_filled['Predict_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    '''
    # Display the preprocessed features
    print("X_train_scaled:")
    print(pd.DataFrame(X_train_scaled, columns=X.columns))
    print("\nX_test_scaled:")
    print(pd.DataFrame(X_test_scaled, columns=X.columns))

    # Display the target variable (y)
    print("\ny_train:")
    print(y_train)
    print("\ny_test:")
    print(y_test)
    '''
    return X_train, X_test, y_train, y_test

# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, stock_prices):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(),
        'Bagging Regressor': BaggingRegressor(),
        'AdaBoost Regressor': AdaBoostRegressor(),
        'Stacking Regressor': StackingRegressor(
            estimators=[
                ('lr', LinearRegression()),
                ('rf', RandomForestRegressor()),
                ('br', BaggingRegressor())
            ],
            final_estimator=Ridge()
        ),
        'Ridge Regression': Ridge(),
        'Moving Average': None
    }
    del models['Moving Average']

    # Store performance metrics and predictions
    results_df = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE'])
    predictions_df = pd.DataFrame(index=stock_prices.index)  # DataFrame to store predictions

    plt.figure(figsize=(12, 6))

    for name, model in models.items():
        print(f"Training and evaluating {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        results_df = results_df.append({'Model': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}, ignore_index=True)

        print(f"Mean Absolute Error (MAE) for {name}: {mae}")
        print(f"Mean Squared Error (MSE) for {name}: {mse}")
        print(f"Root Mean Squared Error (RMSE) for {name}: {rmse}")
        print("\n")
        predictions_df[name] = predictions

    # Plot performance comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='MAE', data=results_df, palette='Blues')
    plt.title('Mean Absolute Error (MAE) Comparison')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='MSE', data=results_df, palette='Blues')
    plt.title('Mean Squared Error (MSE) Comparison')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='RMSE', data=results_df, palette='Blues')
    plt.title('Root Mean Squared Error (RMSE) Comparison')
    plt.show()

    return results_df, predictions_df



# Main function
def main():
    # Fetch stock data
    symbol = 'AAPL'
    start_date = '1970-01-01'
    end_date = '2022-12-31'
    stock_data = fetch_stock_data(symbol, start_date, end_date)
    # Competitor's and Supplier's data
    competitor_symbols = ['^GSPC','^IXIC', '^DJI','^NDX',' XLK','005930.KS', 'MSFT','2354.TW', '002502.SZ','GOOGL','AMZN','IBM','HPQ','INTC', 'CSCO', 'ORCL']  # Add competitor/supplier symbols
    competitor_data = pd.concat([fetch_competitor_stock_data(sym, start_date, end_date) for sym in competitor_symbols], axis=1)
    # Merge stock data with competitor data
    data = pd.merge(stock_data, competitor_data, on='Date', how='inner')
    # Get Additional features
    data = calculate_additional_features(symbol, data)
    # Create 'Predict_Price' column and shift 'Adj Close' values upward by one row
    data['Predict_Price'] = data['Adj Close'].shift(-1)
    data = data.dropna()
    # Data preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(data)
    '''
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(X_test)
    print(y_test)
    '''
    # Train and evaluate models
    results_df, predictions_df = train_and_evaluate_models(X_train, X_test, y_train, y_test, y_test)

    # Sort y_test based on the date
    y_test_sorted = y_test.sort_index()
    predictions_df_sorted = predictions_df.sort_index()
    
    print(results_df)
    # Plot actual values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_sorted.index, y_test_sorted, label='Actual Prices', linewidth=2, color='blue')

    # Plot predicted values for each model
    for column in predictions_df_sorted.columns:
        plt.plot(predictions_df_sorted.index, predictions_df_sorted[column], label=f'{column} Predictions', linestyle='--', alpha=0.8)

    plt.title('Actual Stock Prices vs Model Predictions')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

main()
