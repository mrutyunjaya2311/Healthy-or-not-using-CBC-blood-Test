import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sys

MEDICAL_RANGES = {
    'male': {
        'WBC': (3.4, 9.6),    
        'RBC': (4.35, 5.65),    
        'HGB': (13.2, 16.6),    
        'HCT': (38.3, 48.6),   
        'MCV': (80.0, 96.0),    
        'PLT': (135, 317)       
    },
    'female': {
        'WBC': (3.4, 9.6),     
        'RBC': (3.92, 5.13),    
        'HGB': (11.6, 15.0),    
        'HCT': (35.5, 44.9),   
        'MCV': (80.0, 96.0),   
        'PLT': (157, 371)       
    }
}
def load_real_cbc_data(filepath):
    try:
        data = pd.read_csv(filepath)
        required_columns = list(MEDICAL_RANGES['male'].keys()) + ['gender', 'health_status']
        if not all(col in data.columns for col in required_columns):
            print(f"Error: CSV must contain these columns: {required_columns}")
            return None
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def generate_synthetic_cbc_data(n_samples=1000):
    np.random.seed(42)
    
    data = []
    labels = []
    for gender in ['male', 'female']:
        normal_ranges = MEDICAL_RANGES[gender]
        for _ in range(n_samples // 4): 
            sample = {}
            for param, (lower, upper) in normal_ranges.items():
            
                sample[param] = np.random.uniform(lower + 0.1*(upper-lower), upper - 0.1*(upper-lower))
            sample['gender'] = gender
            data.append(sample)
            labels.append(1)  
        for _ in range(n_samples // 4):  
            sample = {}
            
            abnormal_count = np.random.randint(1, 4)
            abnormal_params = np.random.choice(list(normal_ranges.keys()), abnormal_count, replace=False)
            
            for param, (lower, upper) in normal_ranges.items():
                if param in abnormal_params:
                    if np.random.choice([True, False]):
                        sample[param] = np.random.uniform(max(0, lower - 0.7*(upper-lower)), lower - 0.1*(upper-lower))
                    else:
                        sample[param] = np.random.uniform(upper + 0.1*(upper-lower), upper + 0.7*(upper-lower))
                else:
                    sample[param] = np.random.uniform(lower, upper)
            sample['gender'] = gender
            data.append(sample)
            labels.append(0)
    df = pd.DataFrame(data)
    df['health_status'] = labels
    return df.sample(frac=1).reset_index(drop=True)

def train_cbc_health_model(data=None):
    if data is None:
        print("Generating synthetic CBC data using gender-specific medical ranges")
        df = generate_synthetic_cbc_data()
    else:
        print("Using provided CBC data")
        df = data
    df_processed = pd.get_dummies(df, columns=['gender'], drop_first=True)
    X = df_processed.drop('health_status', axis=1)
    y = df_processed['health_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return clf, scaler, X.columns

def predict_health_status(clf, scaler, features, new_data):
    input_data = new_data.copy()
    gender = input_data.pop('gender', 'male')  

    input_data['gender_male'] = 1 if gender == 'male' else 0
    for feature in features:
        if feature not in input_data:
            input_data[feature] = 0
    input_df = pd.DataFrame([input_data])[features]
    input_scaled = scaler.transform(input_df)
    prediction = clf.predict(input_scaled)[0]
    probability = clf.predict_proba(input_scaled)[0]
    normal_ranges = MEDICAL_RANGES[gender]
    abnormal_indicators = []
    
    for param, value in new_data.items():
        if param in normal_ranges:
            lower, upper = normal_ranges[param]
            if value < lower:
                abnormal_indicators.append(f"{param}: {value:.2f} (low, should be ≥ {lower:.2f})")
            elif value > upper:
                abnormal_indicators.append(f"{param}: {value:.2f} (high, should be ≤ {upper:.2f})")
    result = {
        'health_status': 'Healthy' if prediction == 1 else 'Unhealthy',
        'confidence': probability[prediction],
        'abnormal_indicators': abnormal_indicators
    }
    
    return result

def get_user_input():
    print("\nEnter patient information and CBC values:")
    while True:
        gender = input("Gender (male/female): ").lower()
        if gender in ['male', 'female']:
            break
        print("Please enter either 'male' or 'female'.")
    cbc_values = {'gender': gender}
    normal_ranges = MEDICAL_RANGES[gender]
    
    for param, (lower, upper) in normal_ranges.items():
        while True:
            try:
                value = input(f"{param} ({lower:.1f}-{upper:.1f}): ")
                if value.strip() == '':
                    default_value = (lower + upper) / 2
                    cbc_values[param] = default_value
                    print(f"Using default value: {default_value:.2f}")
                    break
                else:
                    cbc_values[param] = float(value)
                    break
            except ValueError:
                print("Please enter a valid number.")
    return cbc_values
def display_ranges():
    print("\nCBC Reference Ranges:")
    print("=====================")
    params = list(MEDICAL_RANGES['male'].keys())
    param_width = max(len(param) for param in params) + 2
    range_width = 20
    print(f"{'Parameter':<{param_width}}{'Male Range':<{range_width}}{'Female Range':<{range_width}}")
    print("-" * (param_width + range_width * 2))
    for param in params:
        male_lower, male_upper = MEDICAL_RANGES['male'][param]
        female_lower, female_upper = MEDICAL_RANGES['female'][param]
        
        male_range = f"{male_lower:.1f} to {male_upper:.1f}"
        female_range = f"{female_lower:.1f} to {female_upper:.1f}"
        
        print(f"{param:<{param_width}}{male_range:<{range_width}}{female_range:<{range_width}}")
    
    print("\nUnits:")
    print("WBC: billion cells/L")
    print("RBC: trillion cells/L")
    print("HGB: grams/dL")
    print("HCT: percentage (%)")
    print("MCV: femtoliters (fL)")
    print("PLT: billion/L")
    print()

def main():
    print("\n====================================")
    print("CBC Health Analyzer with Gender-Specific Ranges")
    print("====================================\n")
    display_ranges()
    print("\nChoose data source:")
    print("1. Generate synthetic data based on reference ranges")
    print("2. Load data from CSV file")
    choice = input("Enter choice (1/2, default 1): ") or "1"
    
    data = None
    if choice == '2':
        filepath = input("Enter path to CSV file: ")
        data = load_real_cbc_data(filepath)
        if data is None:
            print("Falling back to synthetic data.")
    clf, scaler, features = train_cbc_health_model(data)
    
    while True:
        cbc_values = get_user_input()
        print("\nAnalyzing CBC results...")
        result = predict_health_status(clf, scaler, features, cbc_values)
        
        print(f"\nHealth Assessment: {result['health_status']}")
        print(f"Model Confidence: {result['confidence']:.2%}")
        
        if result['abnormal_indicators']:
            print("\nAbnormal Indicators:")
            for indicator in result['abnormal_indicators']:
                print(f"  • {indicator}")
        else:
            print("\nAll CBC parameters are within normal range.")
        save_option = input("\nWould you like to save these results? (y/n): ").lower()
        if save_option == 'y':
            try:
                filename = input("Enter filename (default: cbc_results.txt): ") or "cbc_results.txt"
                with open(filename, 'w') as f:
                    f.write("CBC Health Assessment Results\n")
                    f.write("===========================\n\n")
                    f.write(f"Gender: {cbc_values['gender'].capitalize()}\n\n")
                    f.write("CBC Values:\n")
                    for param in MEDICAL_RANGES['male'].keys():
                        normal_range = MEDICAL_RANGES[cbc_values['gender']][param]
                        f.write(f"{param}: {cbc_values[param]:.2f} (Reference Range: {normal_range[0]:.1f}-{normal_range[1]:.1f})\n")
                    f.write(f"\nHealth Assessment: {result['health_status']}\n")
                    f.write(f"Model Confidence: {result['confidence']:.2%}\n\n")
                    if result['abnormal_indicators']:
                        f.write("Abnormal Indicators:\n")
                        for indicator in result['abnormal_indicators']:
                            f.write(f"- {indicator}\n")
                    else:
                        f.write("All CBC parameters are within normal range.\n")
                print(f"Results saved to {filename}")
            except Exception as e:
                print(f"Error saving results: {e}")
        another = input("\nWould you like to analyze another set of CBC values? (y/n): ").lower()
        if another != 'y':
            break
    print("\nThank you for using the CBC Health Analyzer!")
if __name__ == "__main__":
    main()