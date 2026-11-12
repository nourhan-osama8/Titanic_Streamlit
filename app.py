from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Titanic Analysis", layout="wide")

ORDERS = {
    "class": ["First", "Second", "Third"],
    "survival": ["Died", "Survived"],
    "age_group": ["child", "teen", "young adult", "adult", "senior"],
}
COLORS = {"Died": "#D95D4A", "Survived": "#0F766E"}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/titanic.csv")
    df.columns = df.columns.str.lower()
    df["class"] = df["pclass"].map({1: "First", 2: "Second", 3: "Third"})
    df["survival"] = df["survived"].map({0: "Died", 1: "Survived"})
    df["port"] = df["embarked"].map(
        {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    )
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["title"] = df["name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    df["age_group"] = pd.cut(
        df["age"],
        [0, 12, 18, 35, 60, 80],
        labels=ORDERS["age_group"],
    )
    return df


df = load_data()

st.title("Titanic Analysis")

if Path("utls/images.jpg").exists():
    st.image("utls/images.jpg")

if st.checkbox("Show sample"):
    st.dataframe(df.head(20), hide_index=True)

classes = st.multiselect(
    "Select classes",
    ORDERS["class"],
    default=ORDERS["class"],
)
gender = st.radio("Select gender", ["All", "male", "female"], horizontal=True)

filtered = df[df["class"].isin(classes)]
if gender != "All":
    filtered = filtered[filtered["sex"] == gender]

if filtered.empty:
    st.warning("No passengers match the selected filters.")
    st.stop()


c1, c2 = st.columns(2)

with c1:
    st.plotly_chart(
        px.histogram(
            filtered,
            x="family_size",
            nbins=11,
            labels={"family_size": "Family size (incl. the passenger)"},
            title="How big were the families on board?",
        ),
        use_container_width=True,
    )

with c2:
    outcome = filtered["survival"].value_counts().reset_index()
    st.plotly_chart(
        px.pie(
            outcome,
            names="survival",
            values="count",
            hole=0.5,
            color="survival",
            color_discrete_map=COLORS,
            category_orders={"survival": ORDERS["survival"]},
            title="Died vs survived",
        ),
        use_container_width=True,
    )

top_titles = filtered["title"].value_counts().head(6).reset_index()
st.plotly_chart(
    px.bar(
        top_titles,
        x="title",
        y="count",
        text_auto=True,
        labels={"title": "Title", "count": "Passengers"},
        title="The 6 most common titles",
    ),
    use_container_width=True,
)