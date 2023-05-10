import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from datetime import datetime
import os
import uuid
from sklearn.preprocessing import OneHotEncoder
import pickle

def logMetadata(x):
    # Check if file exists already exists if not create one:
    metadataCols = ['project', 'run_id', 'date_time_run',
                     'model_name', 'metric_name', 'metric_value']

    z = 'model_metrics.csv'
    model_metrics_path = 'model_metrics/' + z

    if not os.path.isfile(model_metrics_path):
        # Create the CSV file with specified columns
        df = pd.DataFrame(columns=metadataCols)
        df.to_csv(model_metrics_path, index=False)

    # Create temp metadata dataframe using metrics:
    metadata_dict_details = {
        'project': 'example',
        'run_id': str(uuid.uuid4()),
        'date_time_run': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': 'RandomForest'}

    metadata_dict_metrics = {'metric_name': list(x.keys()),
                             'metric_value': list(x.values())
                             }

    metadata_dict = dict(metadata_dict_details, **metadata_dict_metrics)

    # Create temp metadata dataframe:
    metadata_temp = pd.DataFrame(data=metadata_dict, columns=metadataCols)

    # Update metadata csv with temp:
    metadata = pd.read_csv(model_metrics_path, usecols=metadataCols)
    metadata = pd.concat([metadata, metadata_temp])
    metadata.to_csv(model_metrics_path, index=False)

    print(f"Added {len(metadata_temp.index)} rows to metadata file")

    return None

# Load the dataset:
data = pd.read_csv('data/titanic.csv')

# Run data quality checks:
assert len(data.drop_duplicates()) == len(data), "Duplicates present in data"

# Preprocessing:
data.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1, inplace=True)
data = data.dropna()

# Create a OneHotEncoder object:
encoder = OneHotEncoder()

# Fit and transform the "Sex" column using the OneHotEncoder:
sexEncoded = encoder.fit_transform(data[["Sex"]]).toarray()

# Create new column names for the encoded values:
sexCategories = encoder.categories_[0].tolist()
sexColumns = [f"Sex_{category}" for category in sexCategories]

# Convert the encoded values into a DataFrame:
sex_df = pd.DataFrame(data=sexEncoded, columns=sexColumns)

# Append the new columns to the original DataFrame:
data = pd.concat([data.reset_index(drop=True), sex_df], axis=1)
data.drop(['Sex'], axis=1, inplace=True)

# Create the feature matrix and target variable:
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split the dataset into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest Classifier:
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Make predictions on the testing set:
y_pred = rfc.predict(X_test)

# Calculate the model performance metrics:
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))
print('Precision: {:.2f}%'.format(precision*100))
print('Recall: {:.2f}%'.format(recall*100))

# Log metrics:
logMetadata({'accuracy': accuracy,
              'precision': precision,
              'recall': recall})

# Save model:
model_file_name = 'model_rf.pkl'
model_path = 'models/' + model_file_name

os.makedirs(os.path.dirname(model_path), exist_ok=True)

with open(model_path, 'wb') as file:
    pickle.dump(rfc, file)


