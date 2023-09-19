import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings

def read_data(file_path):
    """Read data from an Excel file and return a DataFrame."""
    df = pd.read_excel(file_path)
    return df

def rename_columns(df):
    """Rename DataFrame columns to provide descriptive names."""
    df.columns = ['Relative compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution', 'Heating Load', 'Cooling Load']
    return df

def plot_correlation_heatmap(df):
    """Plot a heatmap to visualize feature correlations."""
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=True, linewidths=.5)
    plt.show()

def prepare_data(df):
    """Prepare feature and target variables for regression."""
    features = ['Relative compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution']
    X = df[features]
    y1 = df['Heating Load']
    y2 = df['Cooling Load']
    return X, y1, y2

def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    return lm

def train_random_forest_regression(X_train, y_train, n_estimators=300):
    """Train a Random Forest Regression model."""
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train.values.ravel())
    return rf

def predict_load(model, input_data):
    """Make load predictions using a trained model."""
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

def save_model(model, filename):
    """Save a trained model to a file using pickle."""
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    """Load a saved model from a file using pickle."""
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def evaluate_model(model, X, y):
    """Evaluate the model's performance using cross-validation."""
    mse = -np.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))
    r2 = np.mean(cross_val_score(model, X, y, scoring='r2', cv=5))
    return mse, r2


if __name__ == '__main__':
    
    # Suppress warnings
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    # Read the data
    df = read_data('energy-efficiency.xlsx')

    # Rename columns for readability
    df = rename_columns(df)

    # Plot feature correlations
    plot_correlation_heatmap(df)

    # Prepare data for regression
    X, y1, y2 = prepare_data(df)

    # Split data into training and testing sets
    X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.33, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.33, random_state=24)

    # Train Linear Regression models
    lm_hot = train_linear_regression(X1_train, y1_train)
    lm_cold = train_linear_regression(X2_train, y2_train)

    # Train Random Forest Regression models
    rf_hot = train_random_forest_regression(X1_train, y1_train)
    rf_cold = train_random_forest_regression(X2_train, y2_train)

    # Save the trained models
    save_model(lm_hot, 'Heating_load_model.sav')
    save_model(lm_cold, 'Cooling_load_model.sav')
    save_model(rf_hot, 'Heating_load_model_rf.sav')
    save_model(rf_cold, 'Cooling_load_model_rf.sav')

    # Load the saved models
    lm_hot = load_model('Heating_load_model.sav')
    lm_cold = load_model('Cooling_load_model.sav')
    rf_hot = load_model('Heating_load_model_rf.sav')
    rf_cold = load_model('Cooling_load_model_rf.sav')

    # Evaluate model performance using cross-validation
    mse_hot_lm, r2_hot_lm = evaluate_model(lm_hot, X, y1)
    mse_cold_lm, r2_cold_lm = evaluate_model(lm_cold, X, y2)
    mse_hot_rf, r2_hot_rf = evaluate_model(rf_hot, X, y1)
    mse_cold_rf, r2_cold_rf = evaluate_model(rf_cold, X, y2)

    print("\nLinear Regression - Heating Load MSE:", mse_hot_lm)
    print("Linear Regression - Heating Load R-squared (R2):", r2_hot_lm)
    print("Linear Regression - Cooling Load MSE:", mse_cold_lm)
    print("Linear Regression - Cooling Load R-squared (R2):", r2_cold_lm)
    
    print("\nRandom Forest Regression - Heating Load MSE:", mse_hot_rf)
    print("Random Forest Regression - Heating Load R-squared (R2):", r2_hot_rf)
    print("Random Forest Regression - Cooling Load MSE:", mse_cold_rf)
    print("Random Forest Regression - Cooling Load R-squared (R2):", r2_cold_rf)

    # Choose the best model based on R2 score
    best_model_heating = lm_hot if r2_hot_lm > r2_hot_rf else rf_hot
    best_model_cooling = lm_cold if r2_cold_lm > r2_cold_rf else rf_cold
    
   # Example input data
    input_data = [0.86, 588, 294, 147, 7, 4, 0, 0]

    # Predict loads using the best models
    prediction_hot = predict_load(best_model_heating, input_data)
    prediction_cold = predict_load(best_model_cooling, input_data)

    print("\nBest Model - Heating Load Prediction:", prediction_hot)
    print("Best Model - Cooling Load Prediction:", prediction_cold)