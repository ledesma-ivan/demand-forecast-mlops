import pandas as pd


def test_save_to_feature_store(tmp_path):
    """Prueba que los datos se guarden en Parquet correctamente"""
    df = pd.DataFrame({"store": [1], "sales": [500]})

    # tmp_path es una carpeta temporal inyectada por pytest
    file_path = tmp_path / "features_v1.parquet"

    # Guardamos y leemos
    df.to_parquet(file_path)
    df_loaded = pd.read_parquet(file_path)

    assert file_path.exists()
    assert df_loaded.shape == (1, 2)
