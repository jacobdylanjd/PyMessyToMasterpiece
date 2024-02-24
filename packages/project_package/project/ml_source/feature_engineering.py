# Install third-party packages:
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns not required for modelling.

    Args:
        df (pd.DataFrame): Dataframe.
    Return:
        df (pd.DataFrame): Dataframe.
    """

    cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    df = df[cols]

    return df


def one_hot_encode_column(df: pd.DataFrame,
                          column: str) -> pd.DataFrame:
    """
    Perform one-hot-encoding for categorical column, replacing original column.

    Args:
        df (pd.DataFrame): Dataframe with categorical column.
        column (str): Categorical column to be one-hot-encoded.
    Return:
        df (pd.DataFrame): Dataframe.
    """

    # Check if the column is categorical
    if df[column].dtype != 'object':
        raise ValueError(f"Column {column} is not categorical") from None

    # Create a OneHotEncoder object:
    encoder = OneHotEncoder()

    # Fit and transform the "Sex" column using the OneHotEncoder
    sex_encoded = encoder.fit_transform(df[[column]]).toarray()

    # Create new column names for the encoded values
    sex_categories = encoder.categories_[0].tolist()
    sex_columns = [f"{column}_{category}" for category in sex_categories]

    # Convert the encoded values into a DataFrame
    sex_df = pd.DataFrame(data=sex_encoded, columns=sex_columns)

    # Append the new columns to the original DataFrame
    df = pd.concat([df.reset_index(drop=True), sex_df], axis=1)
    df = df.drop([column], axis=1)

    return df

