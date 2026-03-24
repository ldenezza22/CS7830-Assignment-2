from pathlib import Path

import polars as pl


def load_dataset() -> pl.DataFrame:
    columns = [
        "ID",
        "Age",
        "Gender",
        "Education",
        "Country",
        "Ethnicity",
        "Nscore",
        "Escore",
        "Oscore",
        "Ascore",
        "Cscore",
        "Impulsive",
        "SS",
        "Alcohol",
        "Amphet",
        "Amyl",
        "Benzos",
        "Caff",
        "Cannabis",
        "Choc",
        "Coke",
        "Crack",
        "Ecstasy",
        "Heroin",
        "Ketamine",
        "Legalh",
        "LSD",
        "Meth",
        "Mushrooms",
        "Nicotine",
        "Semer",
        "VSA",
    ]
    df = pl.read_csv(
        Path("drug_consumption/drug_consumption.data"),
        has_header=False,
        new_columns=columns,
    )
    return df


if __name__ == "__main__":
    df = load_dataset()
    print("hello")
    print(df.head(5))
