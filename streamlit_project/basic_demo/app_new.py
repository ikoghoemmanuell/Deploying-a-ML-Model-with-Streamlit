import streamlit as st
import pandas as pd
import os


# first line after the importation section
st.set_page_config(page_title="Demo app", page_icon="🐞", layout="centered")
DIRPATH = os.path.dirname(os.path.realpath(__file__))


def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


init_df = pd.DataFrame(
    {"petal_length": [], "petal_width": [],
     "sepal_length": [], "sepal_width": [], }
)
# @st.cache_data()  # stop the hot-reload to the function just bellow


def setup(tmp_df_file):
    "Setup the required elements like files, models, global variables, etc"

    # history frame
    if not os.path.exists(tmp_df_file):
        df_history = init_df.copy()
    else:
        df_history = pd.read_csv(tmp_df_file)

    df_history.to_csv(tmp_df_file, index=False)

    return df_history


tmp_df_file = os.path.join(DIRPATH, "tmp", "data_app_dataframe_as_input.csv")
try:
    df_history
except:
    df_history = setup(tmp_df_file)


st.title("🐞 Demo app!")

st.sidebar.write(f"Demo app")
st.sidebar.write(f"This app shows a simple demo of a Streamlit app.")


form = st.form(key="information", clear_on_submit=True)

with form:

    # 👈 An editable dataframe
    edited_df = st.experimental_data_editor(init_df, num_rows="dynamic",)

    submitted = st.form_submit_button(label="Submit")

expander = st.expander("See all records")

if submitted:
    st.success("Thanks!")

    df_history = pd.concat([df_history, edited_df],
                           ignore_index=True).convert_dtypes()
    df_history.to_csv(tmp_df_file, index=False)

    st.balloons()

with expander:

    if submitted:
        st.dataframe(df_history)
        st.download_button(
            "Download this table as CSV",
            convert_df(df_history),
            "file.csv",
            "text/csv",
            key='download-csv'
        )
