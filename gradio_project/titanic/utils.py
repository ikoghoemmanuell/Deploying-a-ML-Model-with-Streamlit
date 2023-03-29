import pandas as pd


def encode_age(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    categories = pd.cut(df.Age, bins, labels=False)
    df.Age = categories
    return df


def encode_fare(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    categories = pd.cut(df.Fare, bins, labels=False)
    df.Fare = categories
    return df


def encode_df(df):
    df = encode_age(df)
    df = encode_fare(df)
    sex_mapping = {"male": 0, "female": 1}
    df = df.replace({"Sex": sex_mapping})
    embark_mapping = {"S": 1, "C": 2, "Q": 3}
    df = df.replace({"Embarked": embark_mapping})
    df.Embarked = df.Embarked.fillna(0)
    df["Company"] = 0
    df.loc[(df["SibSp"] > 0), "Company"] = 1
    df.loc[(df["Parch"] > 0), "Company"] = 2
    df.loc[(df["SibSp"] > 0) & (df["Parch"] > 0), "Company"] = 3
    df = df[
        [
            "PassengerId",
            "Pclass",
            "Sex",
            "Age",
            "Fare",
            "Embarked",
            "Company",
            "Survived",
        ]
    ]
    return df
