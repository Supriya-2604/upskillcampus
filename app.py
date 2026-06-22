import streamlit as st
import pandas as pd
import joblib
import holidays

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Smart City Traffic Forecasting",
    page_icon="🚦",
    layout="wide"
)

# ----------------------------------
# LOAD MODEL
# ----------------------------------
model = joblib.load("notebooks/traffic_model.pkl")

# ----------------------------------
# LOAD DATA
# ----------------------------------
df = pd.read_csv("data/train_aWnotuB.csv")

df['DateTime'] = pd.to_datetime(df['DateTime'])

# ----------------------------------
# HOLIDAY CALENDAR
# ----------------------------------
india_holidays = holidays.India()

# ----------------------------------
# HEADER
# ----------------------------------
st.title("🚦 Smart City Traffic Forecasting Dashboard")

st.markdown("""
### Smart City Traffic Management System

This dashboard predicts traffic volume at city junctions using a Random Forest Machine Learning model.
The system helps authorities anticipate congestion and support infrastructure planning.
""")

# ----------------------------------
# MODEL PERFORMANCE
# ----------------------------------
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", "2.39")
col2.metric("RMSE", "3.54")
col3.metric("R² Score", "0.969")

# ----------------------------------
# SIDEBAR
# ----------------------------------
st.sidebar.header("Prediction Inputs")

junction = st.sidebar.selectbox(
    "Select Junction",
    [1, 2, 3, 4]
)

selected_date = st.sidebar.date_input(
    "Select Date"
)

hour = st.sidebar.slider(
    "Select Hour",
    0,
    23,
    12
)

# ----------------------------------
# FEATURE ENGINEERING
# ----------------------------------
year = selected_date.year
month = selected_date.month
day = selected_date.day

day_of_week = selected_date.weekday()

weekend = 1 if day_of_week >= 5 else 0

is_holiday = 1 if selected_date in india_holidays else 0

# ----------------------------------
# CREATE INPUT DATA
# ----------------------------------
future_data = pd.DataFrame({
    'Junction': [junction],
    'Year': [year],
    'Month': [month],
    'Day': [day],
    'Hour': [hour],
    'DayOfWeek': [day_of_week],
    'Weekend': [weekend],
    'IsHoliday': [is_holiday]
})

# ----------------------------------
# PREDICTION BUTTON
# ----------------------------------
if st.button("🚀 Predict Traffic"):

    prediction = model.predict(future_data)

    traffic = prediction[0]

    st.subheader("Prediction Result")

    st.metric(
        "Predicted Vehicles",
        f"{traffic:.0f}"
    )

    if traffic < 50:
        st.success("🟢 Low Traffic")

    elif traffic < 100:
        st.warning("🟡 Medium Traffic")

    else:
        st.error("🔴 High Traffic")

# ----------------------------------
# TRAFFIC ANALYSIS
# ----------------------------------
st.markdown("---")

st.header("📈 Traffic Analysis")

# Hour Feature
df['Hour'] = df['DateTime'].dt.hour

# Month Feature
df['Month'] = df['DateTime'].dt.month

col1, col2 = st.columns(2)

with col1:

    st.subheader("Average Traffic by Junction")

    junction_avg = (
        df.groupby('Junction')['Vehicles']
        .mean()
    )

    st.bar_chart(junction_avg)

with col2:

    st.subheader("Average Traffic by Hour")

    hourly_avg = (
        df.groupby('Hour')['Vehicles']
        .mean()
    )

    st.line_chart(hourly_avg)

# ----------------------------------
# MONTHLY TREND
# ----------------------------------
st.subheader("Monthly Traffic Trend")

monthly_avg = (
    df.groupby('Month')['Vehicles']
    .mean()
)

st.line_chart(monthly_avg)

# ----------------------------------
# WEEKEND VS WEEKDAY
# ----------------------------------
st.subheader("Weekend vs Weekday Traffic")

df['DayOfWeek'] = df['DateTime'].dt.dayofweek

df['Weekend'] = (
    df['DayOfWeek'] >= 5
).astype(int)

weekend_avg = (
    df.groupby('Weekend')['Vehicles']
    .mean()
)

st.bar_chart(weekend_avg)

# ----------------------------------
# PROJECT INFORMATION
# ----------------------------------
st.markdown("---")

st.header("📋 Project Information")

st.write("""
**Project Title:** Forecasting of Smart City Traffic Patterns

**Objective:** Forecast traffic patterns across four city junctions to help city authorities manage congestion and improve infrastructure planning.

**Machine Learning Model:** Random Forest Regressor

**Features Used:**
- Junction
- Year
- Month
- Day
- Hour
- Day Of Week
- Weekend Indicator
- Holiday Indicator (IsHoliday)

**Key Outcomes:**
- Identified traffic peaks.
- Analyzed traffic trends across four junctions.
- Incorporated holiday effects.
- Generated future traffic forecasts.
- Built an interactive dashboard for decision-making.
""")

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")

st.success("✅ Internship Project Completed Successfully")