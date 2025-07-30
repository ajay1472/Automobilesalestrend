import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class CarSalesPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
    def create_sample_data(self, n_samples=1000):
        """Create sample car sales data for demonstration"""
        np.random.seed(42)
        
        # Generate synthetic car data
        brands = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Nissan']
        fuel_types = ['Petrol', 'Diesel', 'Electric', 'Hybrid']
        transmission = ['Manual', 'Automatic']
        seller_types = ['Dealer', 'Individual']
        
        data = {
            'Brand': np.random.choice(brands, n_samples),
            'Year': np.random.randint(2005, 2024, n_samples),
            'Km_Driven': np.random.randint(1000, 200000, n_samples),
            'Fuel_Type': np.random.choice(fuel_types, n_samples),
            'Transmission': np.random.choice(transmission, n_samples),
            'Owner': np.random.choice(['First', 'Second', 'Third', 'Fourth'], n_samples),
            'Mileage': np.random.uniform(10, 30, n_samples),
            'Engine': np.random.randint(800, 3000, n_samples),
            'Max_Power': np.random.uniform(50, 400, n_samples),
            'Seats': np.random.choice([2, 4, 5, 7, 8], n_samples),
            'Seller_Type': np.random.choice(seller_types, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic selling price based on features
        # Price calculation with realistic factors
        base_price = 500000  # Base price
        
        # Brand factor
        brand_multiplier = {
            'Toyota': 1.0, 'Honda': 1.0, 'Hyundai': 0.8, 'Nissan': 0.9,
            'Ford': 0.85, 'BMW': 2.0, 'Mercedes': 2.2, 'Audi': 1.8
        }
        
        # Calculate selling price
        df['Selling_Price'] = base_price
        
        # Apply brand multiplier
        for brand, multiplier in brand_multiplier.items():
            df.loc[df['Brand'] == brand, 'Selling_Price'] *= multiplier
            
        # Year factor (newer cars cost more)
        df['Selling_Price'] *= (df['Year'] - 2000) / 20
        
        # Km driven factor (more km = lower price)
        df['Selling_Price'] *= (1 - (df['Km_Driven'] / 300000))
        
        # Engine and power factor
        df['Selling_Price'] *= (1 + (df['Engine'] / 5000))
        df['Selling_Price'] *= (1 + (df['Max_Power'] / 1000))
        
        # Fuel type factor
        fuel_multiplier = {'Electric': 1.3, 'Hybrid': 1.2, 'Diesel': 1.1, 'Petrol': 1.0}
        for fuel, multiplier in fuel_multiplier.items():
            df.loc[df['Fuel_Type'] == fuel, 'Selling_Price'] *= multiplier
            
        # Add some noise
        df['Selling_Price'] += np.random.normal(0, 50000, n_samples)
        df['Selling_Price'] = np.abs(df['Selling_Price'])  # Ensure positive prices
        
        return df
    
    def load_and_explore_data(self, df=None):
        """Load and explore the dataset"""
        if df is None:
            # Create sample data if no data provided
            df = self.create_sample_data()
            
        print("=== CAR SALES DATASET OVERVIEW ===")
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nDataset info:")
        print(df.info())
        
        print("\nStatistical summary:")
        print(df.describe())
        
        print("\nMissing values:")
        print(df.isnull().sum())
        
        return df
    
    def visualize_data(self, df):
        """Create visualizations for data exploration"""
        plt.figure(figsize=(20, 15))
        
        # 1. Price distribution
        plt.subplot(3, 4, 1)
        plt.hist(df['Selling_Price'], bins=50, alpha=0.7)
        plt.title('Selling Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        
        # 2. Price vs Year
        plt.subplot(3, 4, 2)
        plt.scatter(df['Year'], df['Selling_Price'], alpha=0.5)
        plt.title('Price vs Year')
        plt.xlabel('Year')
        plt.ylabel('Selling Price')
        
        # 3. Price vs Km Driven
        plt.subplot(3, 4, 3)
        plt.scatter(df['Km_Driven'], df['Selling_Price'], alpha=0.5)
        plt.title('Price vs Km Driven')
        plt.xlabel('Km Driven')
        plt.ylabel('Selling Price')
        
        # 4. Price by Brand
        plt.subplot(3, 4, 4)
        df.groupby('Brand')['Selling_Price'].mean().plot(kind='bar')
        plt.title('Average Price by Brand')
        plt.xticks(rotation=45)
        
        # 5. Price by Fuel Type
        plt.subplot(3, 4, 5)
        df.groupby('Fuel_Type')['Selling_Price'].mean().plot(kind='bar')
        plt.title('Average Price by Fuel Type')
        plt.xticks(rotation=45)
        
        # 6. Price vs Engine
        plt.subplot(3, 4, 6)
        plt.scatter(df['Engine'], df['Selling_Price'], alpha=0.5)
        plt.title('Price vs Engine Size')
        plt.xlabel('Engine (CC)')
        plt.ylabel('Selling Price')
        
        # 7. Price vs Power
        plt.subplot(3, 4, 7)
        plt.scatter(df['Max_Power'], df['Selling_Price'], alpha=0.5)
        plt.title('Price vs Max Power')
        plt.xlabel('Max Power (HP)')
        plt.ylabel('Selling Price')
        
        # 8. Correlation heatmap
        plt.subplot(3, 4, 8)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        
        # 9. Price by Transmission
        plt.subplot(3, 4, 9)
        df.groupby('Transmission')['Selling_Price'].mean().plot(kind='bar')
        plt.title('Average Price by Transmission')
        plt.xticks(rotation=45)
        
        # 10. Price by Owner
        plt.subplot(3, 4, 10)
        df.groupby('Owner')['Selling_Price'].mean().plot(kind='bar')
        plt.title('Average Price by Owner Type')
        plt.xticks(rotation=45)
        
        # 11. Mileage vs Price
        plt.subplot(3, 4, 11)
        plt.scatter(df['Mileage'], df['Selling_Price'], alpha=0.5)
        plt.title('Price vs Mileage')
        plt.xlabel('Mileage (kmpl)')
        plt.ylabel('Selling Price')
        
        # 12. Seats vs Price
        plt.subplot(3, 4, 12)
        df.groupby('Seats')['Selling_Price'].mean().plot(kind='bar')
        plt.title('Average Price by Number of Seats')
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self, df):
        """Preprocess the data for machine learning"""
        print("\n=== DATA PREPROCESSING ===")
        
        # Create feature engineering
        df['Age'] = 2024 - df['Year']
        df['Power_per_CC'] = df['Max_Power'] / df['Engine']
        df['Km_per_Year'] = df['Km_Driven'] / (df['Age'] + 1)  # +1 to avoid division by zero
        
        # Define features and target
        feature_columns = ['Brand', 'Age', 'Km_Driven', 'Fuel_Type', 'Transmission', 
                          'Owner', 'Mileage', 'Engine', 'Max_Power', 'Seats', 
                          'Seller_Type', 'Power_per_CC', 'Km_per_Year']
        
        X = df[feature_columns]
        y = df['Selling_Price']
        
        # Identify categorical and numerical columns
        categorical_features = ['Brand', 'Fuel_Type', 'Transmission', 'Owner', 'Seller_Type']
        numerical_features = ['Age', 'Km_Driven', 'Mileage', 'Engine', 'Max_Power', 
                             'Seats', 'Power_per_CC', 'Km_per_Year']
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ])
        
        self.preprocessor = preprocessor
        self.feature_names = feature_columns
        
        print(f"Features used: {feature_columns}")
        print(f"Target variable: Selling_Price")
        print(f"Dataset shape after preprocessing: {X.shape}")
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple ML models and compare performance"""
        print("\n=== MODEL TRAINING ===")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': pipeline,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"MSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R² Score: {r2:.4f}")
            print(f"CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        best_model = results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best R² Score: {results[best_model_name]['r2']:.4f}")
        
        self.model = best_model
        return results, X_test, y_test
    
    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning for the best model"""
        print("\n=== HYPERPARAMETER TUNING ===")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define parameter grid for Random Forest
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [10, 20, None],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 2]
        }
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        # Grid search
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = grid_search.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Test R² Score: {r2:.4f}")
        print(f"Test RMSE: {rmse:.2f}")
        
        self.model = grid_search.best_estimator_
        return grid_search
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Get feature importance (works for tree-based models)
        if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
            # Get feature names after preprocessing
            categorical_features = ['Brand', 'Fuel_Type', 'Transmission', 'Owner', 'Seller_Type']
            numerical_features = ['Age', 'Km_Driven', 'Mileage', 'Engine', 'Max_Power', 
                                 'Seats', 'Power_per_CC', 'Km_per_Year']
            
            # Get feature names from preprocessor
            ohe_features = self.model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
            all_features = numerical_features + list(ohe_features)
            
            importance = self.model.named_steps['regressor'].feature_importances_
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': all_features,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("Top 15 Most Important Features:")
            print(feature_importance_df.head(15))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            return feature_importance_df
    
    def plot_results(self, results):
        """Plot model comparison and predictions"""
        plt.figure(figsize=(15, 10))
        
        # 1. Model comparison
        plt.subplot(2, 3, 1)
        model_names = list(results.keys())
        r2_scores = [results[name]['r2'] for name in model_names]
        plt.bar(model_names, r2_scores)
        plt.title('Model Comparison (R² Score)')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        
        # 2. RMSE comparison
        plt.subplot(2, 3, 2)
        rmse_scores = [results[name]['rmse'] for name in model_names]
        plt.bar(model_names, rmse_scores)
        plt.title('Model Comparison (RMSE)')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        
        # 3. Actual vs Predicted for best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        best_results = results[best_model_name]
        
        plt.subplot(2, 3, 3)
        plt.scatter(best_results['actual'], best_results['predictions'], alpha=0.5)
        plt.plot([best_results['actual'].min(), best_results['actual'].max()], 
                 [best_results['actual'].min(), best_results['actual'].max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'Actual vs Predicted ({best_model_name})')
        
        # 4. Residuals plot
        plt.subplot(2, 3, 4)
        residuals = best_results['actual'] - best_results['predictions']
        plt.scatter(best_results['predictions'], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        # 5. Cross-validation scores
        plt.subplot(2, 3, 5)
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5)
        plt.title('Cross-Validation Scores')
        plt.ylabel('CV R² Score')
        plt.xticks(rotation=45)
        
        # 6. Prediction error distribution
        plt.subplot(2, 3, 6)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def predict_single_car(self, car_details):
        """Predict price for a single car"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Convert single prediction to DataFrame
        car_df = pd.DataFrame([car_details])
        
        # Add engineered features
        car_df['Age'] = 2024 - car_df['Year']
        car_df['Power_per_CC'] = car_df['Max_Power'] / car_df['Engine']
        car_df['Km_per_Year'] = car_df['Km_Driven'] / (car_df['Age'] + 1)
        
        # Select only the features used in training
        car_df = car_df[self.feature_names]
        
        # Make prediction
        predicted_price = self.model.predict(car_df)[0]
        
        print(f"\n=== PRICE PREDICTION ===")
        print(f"Car Details: {car_details}")
        print(f"Predicted Selling Price: ₹{predicted_price:,.2f}")
        
        return predicted_price
    
    def run_complete_analysis(self):
        """Run the complete car sales prediction analysis"""
        print("🚗 CAR SALES PRICE PREDICTION USING MACHINE LEARNING 🚗")
        print("=" * 60)
        
        # Step 1: Load and explore data
        df = self.load_and_explore_data()
        
        # Step 2: Visualize data
        self.visualize_data(df)
        
        # Step 3: Preprocess data
        X, y = self.preprocess_data(df)
        
        # Step 4: Train models
        results, X_test, y_test = self.train_models(X, y)
        
        # Step 5: Hyperparameter tuning
        self.hyperparameter_tuning(X, y)
        
        # Step 6: Feature importance analysis
        self.feature_importance_analysis()
        
        # Step 7: Plot results
        self.plot_results(results)
        
        # Step 8: Example prediction
        sample_car = {
            'Brand': 'Toyota',
            'Year': 2020,
            'Km_Driven': 25000,
            'Fuel_Type': 'Petrol',
            'Transmission': 'Manual',
            'Owner': 'First',
            'Mileage': 18.5,
            'Engine': 1500,
            'Max_Power': 120,
            'Seats': 5,
            'Seller_Type': 'Dealer'
        }
        
        self.predict_single_car(sample_car)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("The model is now ready for car price predictions!")

# Example usage
if __name__ == "__main__":
    # Create and run the car sales predictor
    predictor = CarSalesPredictor()
    predictor.run_complete_analysis()
    
    # Additional examples
    print("\n" + "="*50)
    print("ADDITIONAL PREDICTIONS")
    print("="*50)
    
    # Example 1: Luxury car
    luxury_car = {
        'Brand': 'BMW',
        'Year': 2022,
        'Km_Driven': 5000,
        'Fuel_Type': 'Petrol',
        'Transmission': 'Automatic',
        'Owner': 'First',
        'Mileage': 12.5,
        'Engine': 2000,
        'Max_Power': 250,
        'Seats': 5,
        'Seller_Type': 'Dealer'
    }
    predictor.predict_single_car(luxury_car)
    
    # Example 2: Budget car
    budget_car = {
        'Brand': 'Hyundai',
        'Year': 2018,
        'Km_Driven': 45000,
        'Fuel_Type': 'Petrol',
        'Transmission': 'Manual',
        'Owner': 'Second',
        'Mileage': 20.0,
        'Engine': 1200,
        'Max_Power': 85,
        'Seats': 5,
        'Seller_Type': 'Individual'
    }
    predictor.predict_single_car(budget_car)
