import gradio as gr
import pandas as pd
from utils import *
import pickle

with open("ml/salary/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("ml/salary/processing.pkl", "rb") as f:
    processor = pickle.load(f)


def predict_salary(
    experience,
    test_score,
    interview_score,
):
    if experience is None:
        return None
    df = pd.DataFrame.from_dict(
        {
            "experience": [experience],
            "test_score": [test_score],
            "interview_score": [interview_score],
        }
    )
    df_ = apply_processing(dataframe=df, **processor)
    pred = model.predict(df_)

    return float(pred[0])


demo = gr.Interface(
    predict_salary,
    [
        gr.Dropdown(
            [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "nine",
                "ten",
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
            ],
            type="value",
            label="Experience",
        ),
        gr.Slider(0, 10, value=5, step=1, label="Technical test score"),
        gr.Slider(0, 10, value=5, step=1, label="Interview score"),
    ],
    gr.Number(label="Predicted Salary"),
    examples=[
        [
            "one",
            1,
            4,
        ],
        [
            "two",
            7,
            4,
        ],
        [
            "four",
            6,
            9,
        ],
    ],
    interpretation="default",
    live=True,
)


demo.launch(
    # share=True,
)
