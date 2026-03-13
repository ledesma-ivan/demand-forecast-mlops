import pandas as pd


def clean_and_merge_data(train_df, features_df, stores_df):
    """Realiza los merges iniciales y limpieza básica."""
    # FÍJATE AQUÍ: Usamos train_df (con _df al final), no train.
    df = pd.merge(train_df, stores_df, on="Store", how="left")
    df = pd.merge(df, features_df, on=["Store", "Date", "IsHoliday"], how="left")

    # Formato de fecha
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def create_features(df):
    """El motor de creación de variables predictivas."""
    data = df.copy()

    # 0. ORDENAR LOS DATOS (CRÍTICO PARA SERIES TEMPORALES)
    data = data.sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)

    # 1. Creando Features Temporales
    data["Week"] = data["Date"].dt.isocalendar().week.astype("int32")
    data["Month"] = data["Date"].dt.month.astype("int32")
    data["Quarter"] = data["Date"].dt.quarter.astype("int32")
    data["Is_Year_End"] = (data["Month"] == 12).astype("int8")

    # 2. Creando Features de Contexto (Markdowns y Feriados)
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    data[markdown_cols] = data[markdown_cols].fillna(0)
    data["Active_Markdowns"] = (data[markdown_cols] > 0).sum(axis=1).astype("int8")
    data["IsHoliday"] = data["IsHoliday"].astype("int8")

    type_map = {"A": 3, "B": 2, "C": 1}
    data["Type_Num"] = data["Type"].map(type_map).astype("int8")

    # 3. Creando Lags (Retardos)
    grouped = data.groupby(["Store", "Dept"])["Weekly_Sales"]
    lags = [1, 2, 4, 8, 52]
    for lag in lags:
        data[f"Sales_Lag_{lag}"] = grouped.shift(lag)

    # 4. Creando Rolling Features
    windows = [4, 8, 12]
    for w in windows:
        data[f"Rolling_Mean_{w}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=w).mean()
        )
        data[f"Rolling_Std_{w}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=w).std()
        )
        data[f"Rolling_Max_{w}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=w).max()
        )

    # 5. Creando Features de Series Cruzadas
    data["Dept_Avg_Sales_All_Stores"] = data.groupby(["Date", "Dept"])[
        "Weekly_Sales"
    ].transform("mean")
    data["Store_Rank_In_Dept"] = data.groupby(["Date", "Dept"])["Weekly_Sales"].rank(
        ascending=False, method="dense"
    )

    return data
