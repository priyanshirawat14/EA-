import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("EA.csv")
    return df

df = load_data()

st.title("👥 HR Attrition Dashboard")
st.markdown("This dashboard helps HR leaders analyze employee attrition patterns and make data-driven decisions.")

# Sidebar filters
st.sidebar.header("📌 Filter the Data")
departments = st.sidebar.multiselect("Department", df['Department'].unique(), default=df['Department'].unique())
genders = st.sidebar.multiselect("Gender", df['Gender'].unique(), default=df['Gender'].unique())
age_range = st.sidebar.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (30, 50))

filtered_df = df[
    (df['Department'].isin(departments)) &
    (df['Gender'].isin(genders)) &
    (df['Age'].between(age_range[0], age_range[1]))
]

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Insights", "📂 Raw Data"])

with tab1:
    st.subheader("1. Employee Attrition Count")
    st.markdown("Shows the overall count of employees who left vs. stayed.")
    st.bar_chart(filtered_df['Attrition'].value_counts())

    st.subheader("2. Attrition by Department")
    st.markdown("Identifies departments with higher attrition rates.")
    dept_attr = filtered_df.groupby("Department")["Attrition"].value_counts().unstack().fillna(0)
    st.bar_chart(dept_attr)

    st.subheader("3. Gender-based Attrition")
    st.markdown("Visualizes attrition by gender.")
    gender_attr = filtered_df.groupby("Gender")["Attrition"].value_counts(normalize=True).unstack()
    st.bar_chart(gender_attr)

    st.subheader("4. KPI Overview")
    st.markdown("Key metrics at a glance.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Age", round(filtered_df["Age"].mean(), 1))
    col2.metric("Average Income", int(filtered_df["MonthlyIncome"].mean()))
    col3.metric("Attrition Rate", f"{round((filtered_df['Attrition'] == 'Yes').mean()*100, 2)}%")

    st.subheader("5. Age Distribution")
    st.markdown("Age spread across all employees.")
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_df['Age'], bins=20, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("6. Pie Chart - Attrition Share")
    st.markdown("Proportion of attrition across the filtered data.")
    attr_counts = filtered_df['Attrition'].value_counts()
    fig2 = px.pie(values=attr_counts, names=attr_counts.index, title="Attrition Ratio")
    st.plotly_chart(fig2)

with tab2:
    st.subheader("7. Monthly Income vs Attrition")
    st.markdown("Box plot showing income differences between employees who stayed vs left.")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=filtered_df, ax=ax3)
    st.pyplot(fig3)

    st.subheader("8. Job Role vs Attrition")
    st.markdown("Which job roles have higher attrition?")
    fig4 = px.histogram(filtered_df, x="JobRole", color="Attrition", barmode="group")
    st.plotly_chart(fig4)

    st.subheader("9. Education Field vs Attrition")
    st.markdown("Attrition by education background.")
    fig5 = px.histogram(filtered_df, x="EducationField", color="Attrition", barmode="group")
    st.plotly_chart(fig5)

    st.subheader("10. Years at Company vs Attrition")
    st.markdown("Experience at company and their attrition status.")
    fig6, ax6 = plt.subplots()
    sns.histplot(data=filtered_df, x="YearsAtCompany", hue="Attrition", multiple="stack", bins=15, ax=ax6)
    st.pyplot(fig6)

    st.subheader("11. Work-Life Balance Impact")
    st.markdown("How work-life balance relates to attrition.")
    fig7 = px.histogram(filtered_df, x="WorkLifeBalance", color="Attrition", barmode="group")
    st.plotly_chart(fig7)

    st.subheader("12. Job Satisfaction vs Attrition")
    st.markdown("Satisfaction level comparison.")
    fig8, ax8 = plt.subplots()
    sns.boxplot(x='Attrition', y='JobSatisfaction', data=filtered_df, ax=ax8)
    st.pyplot(fig8)

    st.subheader("13. Marital Status vs Attrition")
    st.markdown("Attrition trend based on marital status.")
    fig9 = px.histogram(filtered_df, x="MaritalStatus", color="Attrition", barmode="group")
    st.plotly_chart(fig9)

    st.subheader("14. Overtime vs Attrition")
    st.markdown("Effect of overtime on attrition.")
    fig10 = px.histogram(filtered_df, x="OverTime", color="Attrition", barmode="group")
    st.plotly_chart(fig10)

    st.subheader("15. Performance Rating vs Attrition")
    st.markdown("Performance scores and attrition.")
    fig11 = px.box(filtered_df, x="Attrition", y="PerformanceRating", color="Attrition")
    st.plotly_chart(fig11)

    st.subheader("16. Years in Current Role")
    st.markdown("How long people stay in their current role.")
    fig12, ax12 = plt.subplots()
    sns.histplot(data=filtered_df, x="YearsInCurrentRole", hue="Attrition", bins=10, ax=ax12)
    st.pyplot(fig12)

    st.subheader("17. Attrition by Age Group")
    st.markdown("See attrition across binned age ranges.")
    age_bins = pd.cut(filtered_df['Age'], bins=[20, 30, 40, 50, 60])
    fig13 = px.histogram(filtered_df, x=age_bins, color="Attrition", barmode="group")
    st.plotly_chart(fig13)

    st.subheader("18. Heatmap of Correlations")
    st.markdown("See how numerical features relate.")
    num_df = filtered_df.select_dtypes(include='number')
    corr = num_df.corr()
    fig14, ax14 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".1f", cmap='coolwarm', ax=ax14)
    st.pyplot(fig14)

    st.subheader("19. Total Working Years")
    st.markdown("Distribution of employees by total experience.")
    fig15 = px.histogram(filtered_df, x="TotalWorkingYears", color="Attrition", nbins=20)
    st.plotly_chart(fig15)

    st.subheader("20. Education Level Distribution")
    st.markdown("Education level split by attrition.")
    fig16 = px.histogram(filtered_df, x="Education", color="Attrition", barmode="group")
    st.plotly_chart(fig16)

with tab3:
    st.subheader("View Filtered Raw Dataset")
    st.markdown("This is the filtered view of the original dataset based on your selections.")
    st.dataframe(filtered_df)
