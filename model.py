import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class TitanicModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        
    def load_data(self):
        # Load training data
        data_path = os.path.join(os.path.dirname(__file__), 'titanic.csv')
        if not os.path.exists(data_path):
            # Create sample data if file doesn't exist
            self.create_sample_data(data_path)
        
        df = pd.read_csv(data_path)
        return df
    
    def create_sample_data(self, path):
        # Sample Titanic data for demonstration
        data = {
            'PassengerId': range(1, 101),
            'Survived': np.random.choice([0, 1], 100, p=[0.6, 0.4]),
            'Pclass': np.random.choice([1, 2, 3], 100),
            'Name': [f'Passenger {i}' for i in range(1, 101)],
            'Sex': np.random.choice(['male', 'female'], 100),
            'Age': np.random.normal(30, 15, 100).clip(1, 80),
            'SibSp': np.random.choice([0, 1, 2, 3], 100),
            'Parch': np.random.choice([0, 1, 2], 100),
            'Ticket': [f'TICKET{i}' for i in range(1, 101)],
            'Fare': np.random.exponential(30, 100),
            'Cabin': [f'Cabin{i}' if np.random.random() > 0.7 else '' for i in range(1, 101)],
            'Embarked': np.random.choice(['M', 'CH', 'K', 'KO'], 100)
        }
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
    
    def preprocess_data(self, df):
        # Handle missing values
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Embarked'].fillna('S', inplace=True)
        
        # Encode categorical variables
        for col in ['Sex', 'Embarked']:
            if col not in self.label_encoders:
                # Create new encoder with all possible values
                if col == 'Embarked':
                    all_embarked = ['M', 'CH', 'K', 'KO']
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(all_embarked)
                else:
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(df[col])
            
            # Handle unknown labels by mapping to most common or default
            try:
                df[col] = self.label_encoders[col].transform(df[col])
            except ValueError:
                # For unknown labels, map to default (Southampton for embarked)
                if col == 'Embarked':
                    df[col] = self.label_encoders[col].transform(['S'] * len(df))
                else:
                    # For other columns, use the first known class
                    known_classes = list(self.label_encoders[col].classes_)
                    df[col] = self.label_encoders[col].transform([known_classes[0]] * len(df))
        
        return df
    
    def train_model(self):
        df = self.load_data()
        df = self.preprocess_data(df)
        
        X = df[self.features]
        y = df['Survived']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(self.model, 'titanic_model.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        
        return self.model.score(X_test, y_test)
    
    def load_model(self):
        try:
            self.model = joblib.load('titanic_model.pkl')
            self.label_encoders = joblib.load('label_encoders.pkl')
            return True
        except:
            return False
    
    def predict_survival(self, passenger_data):
        if not self.model:
            if not self.load_model():
                self.train_model()
        
        # Preprocess input
        input_df = pd.DataFrame([passenger_data])
        input_df = self.preprocess_data(input_df)
        
        prediction = self.model.predict(input_df[self.features])[0]
        probability = self.model.predict_proba(input_df[self.features])[0][1]
        
        return {
            'survived': bool(prediction),
            'probability': float(probability),
            'confidence': 'High' if probability > 0.7 else 'Medium' if probability > 0.5 else 'Low'
        }

# Global model instance
titanic_model = TitanicModel()