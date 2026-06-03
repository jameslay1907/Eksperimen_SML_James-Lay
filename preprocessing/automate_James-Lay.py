from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(data, target_column):
    # Menentukan fitur numerik dan kategorikal
    data = data.drop(columns=['Student_ID'])
    numeric_features  = data.select_dtypes(include=['float64', 'int64']).columns.to_list()
    categorical_features = data.select_dtypes(include=['object']).columns.to_list()
    
    if target_column in numeric_features:
        numeric_features.remove(target_column)

    if target_column in categorical_features:
        categorical_features.remove(target_column)
    
    # Outlier
    Q1 = data[numeric_features].quantile(0.25)
    Q3 = data[numeric_features].quantile(0.75)
    IQR = Q3 - Q1

    condition = ~((data[numeric_features] < (Q1 - 1.5 * IQR)) | (data[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)

    data = data.loc[condition].copy()

    # Encoding
    label_encoder = LabelEncoder()

    for column in categorical_features:
        data[column] = label_encoder.fit_transform(data[column])

    target_column_mapping = {"Low":0, "Medium":1, "High":2}

    data[target_column] = data[target_column].map(target_column_mapping)

    # Split Data
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scaling Numerical Features
    scaler = StandardScaler()

    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = pd.read_csv('ai_student_impact_raw.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data, 'Burnout_Risk_Level')
    save_path = 'preprocessing/ai_student_impact_preprocessing'
    X_train.to_csv(f'{save_path}/X_train.csv', index=False)
    X_test.to_csv(f'{save_path}/X_test.csv', index=False)
    y_train.to_csv(f'{save_path}/y_train.csv', index=False)
    y_test.to_csv(f'{save_path}/y_test.csv', index=False)
    
