# Install third-party packages:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def model_training(data: dict) -> dict:
    """
    Run supervised learning model training and testing.

    Args:
        data (dict): Data split for model train/test.
    Return:
        out (dict): Trained model and associated metadata.
    """

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    # Train the Random Forest Classifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the testing set:
    y_pred = model.predict(X_test)

    # Calculate the model performance metrics:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    out = {'model': model,
           'accuracy': accuracy,
           'precision': precision,
           'recall': recall}

    return out


def split_data(df: pd.DataFrame) -> dict:
    """
    Split data into train and test datasets.

    Args:
        df (pd.DataFrame): Data to be split for model train/test.
    Return:
        data (dict): Data split for model train/test.
    """

    # Create the feature df and target variable:
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Split the dataset into training and testing sets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    data = {'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test}

    return data

