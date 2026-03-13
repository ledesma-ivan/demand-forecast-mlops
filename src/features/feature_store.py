import os

import numpy as np
import pandas as pd


class LocalFeatureStore:
    def __init__(self, base_dir="data/processed/walmart_features"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            print(f"Directorio creado: {self.base_dir}")

    def _sanitize_types(self, df):
        """Limpia los tipos de datos incompatibles con PyArrow antes de guardar"""
        df_clean = df.copy()

        tipos_texto = ["object", "category", "str"]

        for col in df_clean.select_dtypes(include=tipos_texto).columns:
            df_clean[col] = df_clean[col].astype(str)

        if "Week" in df_clean.columns:
            df_clean["Week"] = np.int32(df_clean["Week"])

        return df_clean

    def save_features(self, df, feature_group_name):
        path = os.path.join(self.base_dir, f"{feature_group_name}.parquet")

        df_safe = self._sanitize_types(df)

        # CAMBIO AQUÍ: Usamos pyarrow en lugar de fastparquet
        df_safe.to_parquet(path, engine="pyarrow", index=False)
        print(f"✅ Features guardadas exitosamente en: {path}")

    def load_features(self, feature_group_name):
        path = os.path.join(self.base_dir, f"{feature_group_name}.parquet")
        if os.path.exists(path):
            print(f"📥 Cargando features desde: {path}")

            # CAMBIO AQUÍ: Usamos pyarrow en lugar de fastparquet
            return pd.read_parquet(path, engine="pyarrow")
        else:
            raise FileNotFoundError(f"No se encontró el archivo: {path}")
