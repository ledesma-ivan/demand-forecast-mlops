# tests/test_features.py
import pandas as pd
import pytest

# from src.features import create_lags_and_rolling  <-- Tu función real


@pytest.fixture
def sample_sales_for_lags():
    """Un mini dataset de 3 semanas para la misma tienda y departamento"""
    return pd.DataFrame(
        {
            "store": [1, 1, 1],
            "dept": [1, 1, 1],
            "date": pd.to_datetime(["2010-01-01", "2010-01-08", "2010-01-15"]),
            "sales": [100, 150, 200],
        }
    )


def test_lags_generation(sample_sales_for_lags):
    """Verifica que el lag de 1 semana se desplace correctamente"""

    # df_featured = create_lags_and_rolling(sample_sales_for_lags)
    # Simulo tu función real:
    df = sample_sales_for_lags.copy()
    df["sales_lag_1"] = df["sales"].shift(1)

    # El primer registro debería ser NaN porque no hay semana previa
    assert pd.isna(df["sales_lag_1"].iloc[0])

    # El segundo registro debería tener las ventas de la semana 1 (100)
    assert df["sales_lag_1"].iloc[1] == 100
