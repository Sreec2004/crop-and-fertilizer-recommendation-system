import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import joblib

# Load the dataset
data = pd.read_csv("Crop and fertilizer dataset.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# Identify and exclude constant variables
constant_vars = [col for col in data.columns if data[col].nunique() == 1]
data.drop(columns=constant_vars, inplace=True)

# Ensure target variables are factors
data['Crop_name'] = data['Crop_name'].astype('category')
data['Fertilizer'] = data['Fertilizer'].astype('category')
data['Link'] = data['Link'].astype('category')

# Save the crop names, fertilizer names, and links mapping
crop_names = dict(enumerate(data['Crop_name'].cat.categories))
fertilizer_names = dict(enumerate(data['Fertilizer'].cat.categories))
links = dict(enumerate(data['Link'].cat.categories))
joblib.dump(crop_names, 'crop_names.joblib')
joblib.dump(fertilizer_names, 'fertilizer_names.joblib')
joblib.dump(links, 'links.joblib')

# Encode crop names, fertilizer names, and links
data['Crop_name'] = data['Crop_name'].cat.codes
data['Fertilizer'] = data['Fertilizer'].cat.codes
data['Link'] = data['Link'].cat.codes

# One-hot encode categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Normalize numerical features
numeric_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']
scaler = StandardScaler()
data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Save the feature names
feature_names = data_encoded.drop(columns=['Crop_name', 'Fertilizer', 'Link']).columns
joblib.dump(feature_names, 'feature_names.joblib')

# Split the data into training and testing sets
X = data_encoded.drop(columns=['Crop_name', 'Fertilizer', 'Link'])
y_crop = data_encoded['Crop_name']
y_fertilizer = data_encoded['Fertilizer']
y_link = data_encoded['Link']

X_train, X_test, y_crop_train, y_crop_test, y_fertilizer_train, y_fertilizer_test, y_link_train, y_link_test = train_test_split(
    X, y_crop, y_fertilizer, y_link, test_size=0.3, random_state=123)

# Train the crop prediction model
crop_estimators = [
    ('nb', GaussianNB()),
    ('tree', DecisionTreeClassifier(random_state=123)),
    ('svc', SVC(kernel='linear', random_state=123, probability=True))
]
crop_stacking_clf = StackingClassifier(
    estimators=crop_estimators,
    final_estimator=LogisticRegression(max_iter=1000)
)
crop_stacking_clf.fit(X_train, y_crop_train)
joblib.dump(crop_stacking_clf, 'crop_stacking_classifier.joblib')

# Train the fertilizer prediction model
fertilizer_estimators = [
    ('nb', GaussianNB()),
    ('tree', DecisionTreeClassifier(random_state=123)),
    ('svc', SVC(kernel='linear', random_state=123, probability=True))
]
fertilizer_stacking_clf = StackingClassifier(
    estimators=fertilizer_estimators,
    final_estimator=LogisticRegression(max_iter=1000)
)
fertilizer_stacking_clf.fit(X_train, y_fertilizer_train)
joblib.dump(fertilizer_stacking_clf, 'fertilizer_stacking_classifier.joblib')

# Train the link prediction model
link_estimators = [
    ('nb', GaussianNB()),
    ('tree', DecisionTreeClassifier(random_state=123)),
    ('svc', SVC(kernel='linear', random_state=123, probability=True))
]
link_stacking_clf = StackingClassifier(
    estimators=link_estimators,
    final_estimator=LogisticRegression(max_iter=1000)
)
link_stacking_clf.fit(X_train, y_link_train)
joblib.dump(link_stacking_clf, 'link_stacking_classifier.joblib')
