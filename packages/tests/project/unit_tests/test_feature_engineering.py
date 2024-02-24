# Install third-party packages:
import pytest

# Install project packages:
from ....project_package.project.ml_source.feature_engineering import drop_columns, one_hot_encode_column
from ..data.test_data import sample_dataframe


@pytest.mark.unit
def test_drop_columns(sample_dataframe):

    result_df = drop_columns(sample_dataframe)

    expected_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

    assert list(result_df.columns) == list(expected_columns)


@pytest.mark.unit
def test_one_hot_encode_column(sample_dataframe):

    result_df = one_hot_encode_column(sample_dataframe, 'Sex')

    expected_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket',
                        'Fare', 'Cabin', 'Embarked', 'Sex_male', 'Sex_female']

    assert 'Sex' not in list(result_df.columns)
    assert set(result_df.columns) == set(expected_columns)

    with pytest.raises(ValueError, match=r"Column .+ is not categorical"):
        one_hot_encode_column(sample_dataframe, 'Fare')
