import pandas as pd

# df = pd.read_csv(
#     "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
# )
# print((df["sepal_length"] > 6.0).sum())

# print(df.groupby("species")["petal_length"].mean().idxmax())

df_mpg = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
)
# print(round(df["horsepower"].corr(df["mpg"]), 3))


df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
)
print(df.groupby("year")["passengers"].sum().idxmax())

df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
)
print(((df["time"] == "Dinner") & (df["size"] > 4)).sum())


print(df_mpg.nlargest(3, "mpg")["name"].tolist())
