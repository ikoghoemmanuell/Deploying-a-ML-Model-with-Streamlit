import streamlit as st
import pandas as pd
import os


# first line after the importation section
st.set_page_config(page_title="Demo app", page_icon="🐞", layout="centered")
DIRPATH = os.path.dirname(os.path.realpath(__file__))


@st.cache_resource()  # stop the hot-reload to the function just bellow
def setup(tmp_df_file):
    "Setup the required elements like files, models, global variables, etc"
    pd.DataFrame(
        dict(
            firstname=[],
            laststname=[],
            gender=[],
            special_date=[],
            comment=[],
            height=[],
        )
    ).to_csv(tmp_df_file, index=False)


tmp_df_file = os.path.join(DIRPATH, "tmp", "data.csv")
setup(tmp_df_file)


st.title("🐞 Demo app!")

st.sidebar.write(f"Demo app")
st.sidebar.write(f"This app shows a simple demo of a Streamlit app.")

form = st.form(key="information", clear_on_submit=True)

with form:

    cols = st.columns((1, 1))
    firstname = cols[0].text_input("Firstname")
    laststname = cols[1].text_input("Lastname")
    gender = cols[0].selectbox("Gender:", ["Male", "Female", "Robot", "Other"], index=2)
    special_date = cols[1].date_input("Anniversary")
    comment = st.text_area("Comment:")
    cols = st.columns(2)

    height = cols[1].slider("How tall are you in meter (m)? :", 0.2, 4.0, 1.30)
    submitted = st.form_submit_button(label="Submit")

if submitted:
    st.success("Thanks!")
    pd.read_csv(tmp_df_file).append(
        dict(
            firstname=firstname,
            laststname=laststname,
            gender=gender,
            special_date=special_date,
            comment=comment,
            height=height,
        ),
        ignore_index=True,
    ).to_csv(tmp_df_file, index=False)
    st.balloons()

expander = st.expander("See all records")
with expander:
    df = pd.read_csv(tmp_df_file)
    st.dataframe(df)
