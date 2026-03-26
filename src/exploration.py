import polars as pl
import numpy as np
import matplotlib.pyplot as plot
from pathlib import Path
import seaborn as sns
import os


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


df = load_dataset()
print(df.head(5))

os.makedirs(Path("figures"), exist_ok=True)
df.describe().write_csv(Path("figures/descriptive_statistics.csv"))

os.makedirs(Path("figures/frequencies"), exist_ok=True)


def save_frequencies_to_csv(column: str, df: pl.DataFrame):
    df.select(pl.col(column).value_counts()).unnest(column).write_csv(
        Path(f"figures/frequencies/{column}.csv")
    )


save_frequencies_to_csv("Alcohol", df)
save_frequencies_to_csv("Cannabis", df)
save_frequencies_to_csv("Mushrooms", df)
save_frequencies_to_csv("Nicotine", df)
save_frequencies_to_csv("LSD", df)

os.makedirs(Path("figures/correlations"), exist_ok=True)
sns.pairplot(df.to_pandas())
plot.show()  # ("figures/correlations/coorelation_matrix.jpg")
heatmap_df = df.select(
    [
        pl.col("Age"),
        pl.col("Gender"),
        pl.col("Education"),
        pl.col("Country"),
        pl.col("Ethnicity"),
        pl.col("Nscore"),
        pl.col("Escore"),
        pl.col("Oscore"),
        pl.col("Ascore"),
        pl.col("Cscore"),
        pl.col("Impulsive"),
        pl.col("SS"),
    ]
)
plot.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_df.to_pandas().corr(),
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
)
plot.show()  # savefig(Path("figures/correlations/heatmap.jpg"))
