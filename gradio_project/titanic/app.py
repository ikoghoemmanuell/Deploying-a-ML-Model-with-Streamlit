import gradio as gr
import pandas as pd
from utils import *
import pickle

with open("ml/titanic/model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_survival(passenger_class, is_male, age, company, fare, embark_point):
    if passenger_class is None or embark_point is None:
        return None
    df = pd.DataFrame.from_dict(
        {
            "Pclass": [passenger_class + 1],
            "Sex": [0 if is_male else 1],
            "Age": [age],
            "Company": [
                (1 if "Sibling" in company else 0) + (2 if "Child" in company else 0)
            ],
            "Fare": [fare],
            "Embarked": [embark_point + 1],
        }
    )
    df = encode_age(df)
    df = encode_fare(df)
    pred = model.predict_proba(df)[0]
    return {"Perishes": float(pred[0]), "Survives": float(pred[1])}


demo = gr.Interface(
    predict_survival,
    [
        gr.Dropdown(["first", "second", "third"], type="index"),
        "checkbox",
        gr.Slider(0, 80, value=25, step=1),
        gr.CheckboxGroup(["Sibling", "Child"], label="Travelling with (select all)"),
        gr.Number(value=20),
        gr.Radio(["S", "C", "Q"], type="index"),
    ],
    "label",
    examples=[
        ["first", True, 30, [], 50, "S"],
        ["second", False, 40, ["Sibling", "Child"], 10, "Q"],
        ["third", True, 30, ["Child"], 20, "S"],
    ],
    interpretation="default",
    live=True,
)


demo.launch(
    # share=True,
)
