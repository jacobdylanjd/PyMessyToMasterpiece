# Install third-party packages:
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_dataframe():
    d_test = {'PassengerId': [1, 2, 3, 4, 5, 6, 7, 8, 9],
              'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1],
              'Pclass': [3, 1, 3, 1, 3, 3, 1, 2, 1],
              'Name': ['Futrelle, Mrs. Jacques Heath (Lily May Peel)', 'Allen, Mr. William Henry',
                       'Moran, Mr. James', 'McCarthy, Mr. Timothy J', 'Palsson, Master. Gosta Leonard',
                       'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)', 'Nasser, Mrs. Nicholas (Adele Achem)',
                       'Sandstrom, Miss. Marguerite Rut', 'Bonnell, Miss. Elizabeth'],
              'Sex': ['female', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'female'],
              'Age': [20, 38, 23, 45, 2, 5, 32, 23, 7],
              'SibSp': [0, 1, 0, 0, 4, 0, 1, 1, 0],
              'Parch': [0, 5, 0, 0, 1, 0, 0, 2, 1],
              'Ticket': ['113803', '373450', '330877', '17463', '113783', '347082', '350406', '347077', '2631'],
              'Fare': [8.05, 8.4583, 51.8625, 11.1333, 23.22, 34.055, 34.70, 27.0, 36.98],
              'Cabin': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', np.nan],
              'Embarked': ['Q', 'B', 'Q', 'D', 'X', 'F', 'X', 'H', np.nan]
              }

    return pd.DataFrame(d_test)
