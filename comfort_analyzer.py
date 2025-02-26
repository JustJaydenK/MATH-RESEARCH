import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import random  # Moved to top
from datetime import datetime, timedelta  # Moved to top

# Set page configuration
st.set_page_config(
    page_title="Davao City NHS Student Comfort Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to generate sample data (fixed imports)
@st.cache_data
def generate_sample_data():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define the number of respondents
    num_respondents = 50
    
    # 1. Physical Environment Variables
    classroom_temperatures = np.random.uniform(25, 34, num_respondents).round(1)  # in Celsius
    noise_levels = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    cleanliness_ratings = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    seating_comfort = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    lighting_quality = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    
    # 2. Psychological Variables
    stress_levels = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    sense_of_belonging = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    perceived_safety = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    
    # 3. Social Variables
    teacher_relationship_quality = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    peer_relationship_quality = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    
    # 4. Demographic Variables
    age_values = np.random.randint(12, 19, num_respondents)  # Ages 12-18
    grade_levels = [random.choice([7, 8, 9, 10, 11, 12]) for _ in range(num_respondents)]
    gender_options = ['Male', 'Female', 'Prefer not to say']
    gender_values = [random.choice(gender_options) for _ in range(num_respondents)]
    
    # 5. Academic Variables
    study_hours_per_week = np.random.randint(1, 31, num_respondents)  # 1-30 hours
    academic_performance = np.random.randint(70, 100, num_respondents)  # 70-99 grade average
    
    # 6. School Facility Variables
    facility_satisfaction_options = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
    library_satisfaction = [random.choice(facility_satisfaction_options) for _ in range(num_respondents)]
    cafeteria_satisfaction = [random.choice(facility_satisfaction_options) for _ in range(num_respondents)]
    bathroom_cleanliness = [random.choice(facility_satisfaction_options) for _ in range(num_respondents)]
    computer_lab_access = [random.choice(facility_satisfaction_options) for _ in range(num_respondents)]
    
    # 7. Temporal Variables
    commute_time_minutes = np.random.randint(5, 121, num_respondents)  # 5-120 minutes
    start_date = datetime(2024, 1, 1)
    survey_dates = [start_date + timedelta(days=random.randint(0, 60)) for _ in range(num_respondents)]
    survey_dates = [date.strftime('%Y-%m-%d') for date in survey_dates]
    
    # 8. Resource Variables
    textbook_availability = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    internet_access = [random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Always']) for _ in range(num_respondents)]
    study_space_availability = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    
    # 9. Extracurricular Variables
    extracurricular_participation = [random.choice(['None', '1 activity', '2 activities', '3+ activities']) for _ in range(num_respondents)]
    extracurricular_satisfaction = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    
    # 10. Overall Satisfaction
    overall_school_satisfaction = np.random.randint(1, 11, num_respondents)  # Scale 1-10
    
    # Create a DataFrame
    data = {
        # Physical Environment
        'Classroom_Temperature_Celsius': classroom_temperatures,
        'Noise_Level_1to10': noise_levels,
        'Cleanliness_Rating_1to10': cleanliness_ratings,
        'Seating_Comfort_1to10': seating_comfort,
        'Lighting_Quality_1to10': lighting_quality,
        
        # Psychological 
        'Stress_Level_1to10': stress_levels,
        'Sense_of_Belonging_1to10': sense_of_belonging,
        'Perceived_Safety_1to10': perceived_safety,
        
        # Social
        'Teacher_Relationship_Quality_1to10': teacher_relationship_quality,
        'Peer_Relationship_Quality_1to10': peer_relationship_quality,
        
        # Demographic
        'Age': age_values,
        'Grade_Level': grade_levels,
        'Gender': gender_values,
        
        # Academic
        'Weekly_Study_Hours': study_hours_per_week,
        'Academic_Performance_Grade': academic_performance,
        
        # School Facilities
        'Library_Satisfaction': library_satisfaction,
        'Cafeteria_Satisfaction': cafeteria_satisfaction,
        'Bathroom_Cleanliness_Satisfaction': bathroom_cleanliness,
        'Computer_Lab_Access_Satisfaction': computer_lab_access,
        
        # Temporal
        'Commute_Time_Minutes': commute_time_minutes,
        'Survey_Date': survey_dates,
        
        # Resources
        'Textbook_Availability_1to10': textbook_availability,
        'Internet_Access_Frequency': internet_access,
        'Study_Space_Availability_1to10': study_space_availability,
        
        # Extracurricular
        'Extracurricular_Participation': extracurricular_participation,
        'Extracurricular_Satisfaction_1to10': extracurricular_satisfaction,
        
        # Overall Satisfaction
        'Overall_School_Satisfaction_1to10': overall_school_satisfaction
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add student ID
    df['Student_ID'] = ['DCNHS' + str(i+1).zfill(3) for i in range(num_respondents)]
    
    # Reorder columns to put Student_ID first
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    
    return df

# Import needed modules
import random
from datetime import datetime, timedelta

# Generate or load data
df = generate_sample_data()

# Define categories for rating scales
def categorize_rating(value):
    if value <= 3:
        return "Low"
    elif value <= 7:
        return "Medium"
    else:
        return "High"

# Apply categorization to rating columns
for col in df.columns:
    if '1to10' in col:
        df[f'{col}_Category'] = df[col].apply(categorize_rating)

# Sidebar for navigation
st.sidebar.title("Davao City NHS")
st.sidebar.image("https://via.placeholder.com/150?text=DCNHS+Logo", width=150)

# Navigation options
page = st.sidebar.radio("Navigation", 
    ["Dashboard", "Frequency Tables", "Cross-Tabulations", 
    "Advanced Visualizations", "Predictive Analytics", "Cluster Analysis", 
    "Recommendations", "Raw Data"])

# Function to create frequency tables
def create_frequency_table(df, column):
    freq = df[column].value_counts().reset_index()
    freq.columns = [column, 'Frequency']
    freq['Percentage'] = (freq['Frequency'] / freq['Frequency'].sum() * 100).round(2)
    freq['Cumulative Percentage'] = freq['Percentage'].cumsum().round(2)
    return freq

# Function to download table as CSV
def download_csv(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'
    return href

# Main Dashboard
if page == "Dashboard":
    st.title("Student Comfort Analytics Dashboard")
    st.subheader("Enhancing Youth Experiences at Davao City National High School")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Satisfaction", f"{df['Overall_School_Satisfaction_1to10'].mean():.2f}/10")
    
    with col2:
        st.metric("Number of Students", f"{len(df)}")
    
    with col3:
        high_satisfaction = (df['Overall_School_Satisfaction_1to10'] >= 8).sum()
        st.metric("High Satisfaction Students", f"{high_satisfaction} ({high_satisfaction/len(df)*100:.1f}%)")
    
    # Summary statistics by category
    st.subheader("Summary by Category")
    category_cols = {
        "Physical Environment": ["Classroom_Temperature_Celsius", "Noise_Level_1to10", 
                                "Cleanliness_Rating_1to10", "Seating_Comfort_1to10", 
                                "Lighting_Quality_1to10"],
        "Psychological Factors": ["Stress_Level_1to10", "Sense_of_Belonging_1to10", 
                                "Perceived_Safety_1to10"],
        "Social Factors": ["Teacher_Relationship_Quality_1to10", "Peer_Relationship_Quality_1to10"],
        "Academic": ["Weekly_Study_Hours", "Academic_Performance_Grade"]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_category = st.selectbox("Select Category", list(category_cols.keys()))
        
    with col2:
        chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Radar Chart", "Histogram"])
    
    selected_cols = category_cols[selected_category]
    
    if chart_type == "Bar Chart":
        fig = px.bar(
            df[selected_cols].mean().reset_index().rename(columns={0:'Average', 'index':'Metric'}),
            x='Metric', y='Average',
            title=f"Average {selected_category} Metrics",
            color='Average',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Radar Chart":
        categories = [col.split('_')[0] for col in selected_cols]
        values = df[selected_cols].mean().tolist()
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=selected_category
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=False,
            title=f"{selected_category} Profile"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Histogram
        for col in selected_cols:
            fig = px.histogram(df, x=col, 
                title=f"Distribution of {col}",
                color_discrete_sequence=['#3366CC'])
            st.plotly_chart(fig, use_container_width=True)
    
    # Grade level distribution
    st.subheader("Student Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(df, names='Gender', title='Gender Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='Grade_Level', 
                        title='Grade Level Distribution',
                        color_discrete_sequence=['#FF6633'])
        st.plotly_chart(fig, use_container_width=True)

# Frequency Tables
elif page == "Frequency Tables":
    st.title("Frequency Distribution Tables")
    
    # Tab selection for different types of variables
    tab1, tab2, tab3 = st.tabs(["Categorical Variables", "Rating Scales", "Numerical Variables"])
    
    with tab1:
        st.header("Categorical Variables")
        categorical_cols = ['Gender', 'Grade_Level', 'Extracurricular_Participation', 
                        'Library_Satisfaction', 'Cafeteria_Satisfaction', 
                        'Bathroom_Cleanliness_Satisfaction', 'Computer_Lab_Access_Satisfaction',
                        'Internet_Access_Frequency']
        
        selected_cat_col = st.selectbox("Select Categorical Variable", categorical_cols)
        
        freq_table = create_frequency_table(df, selected_cat_col)
        st.table(freq_table)
        
        st.markdown(download_csv(freq_table, f"frequency_{selected_cat_col}"), unsafe_allow_html=True)
        
        fig = px.bar(freq_table, x=selected_cat_col, y='Frequency', 
                title=f"Frequency Distribution of {selected_cat_col}",
                text='Percentage')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Rating Scale Variables (1-10)")
        rating_cols = [col for col in df.columns if '1to10' in col and 'Category' not in col]
        
        selected_rating_col = st.selectbox("Select Rating Scale Variable", rating_cols)
        
        # Group into categories for display
        rating_cat_col = f"{selected_rating_col}_Category"
        
        freq_table = create_frequency_table(df, rating_cat_col)
        st.table(freq_table)
        
        st.markdown(download_csv(freq_table, f"frequency_{rating_cat_col}"), unsafe_allow_html=True)
        
        # Plot both the original distribution and the categorized one
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=selected_rating_col, nbins=10,
                            title=f"Distribution of {selected_rating_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(df, names=rating_cat_col, 
                    title=f"Categories of {selected_rating_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Numerical Variables")
        numerical_cols = ['Age', 'Weekly_Study_Hours', 'Academic_Performance_Grade', 
                        'Commute_Time_Minutes', 'Classroom_Temperature_Celsius']
        
        selected_num_col = st.selectbox("Select Numerical Variable", numerical_cols)
        
        # Create bins for the selected numerical variable
        if selected_num_col == 'Age':
            bins = list(range(12, 20))
            labels = [f"{i}" for i in range(12, 19)]
        elif selected_num_col == 'Weekly_Study_Hours':
            bins = [0, 5, 10, 15, 20, 25, 30]
            labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30']
        elif selected_num_col == 'Academic_Performance_Grade':
            bins = [70, 75, 80, 85, 90, 95, 100]
            labels = ['70-74', '75-79', '80-84', '85-89', '90-94', '95-99']
        elif selected_num_col == 'Commute_Time_Minutes':
            bins = [0, 15, 30, 45, 60, 90, 120]
            labels = ['0-15', '16-30', '31-45', '46-60', '61-90', '91-120']
        else:  # Classroom_Temperature_Celsius
            bins = [25, 27, 29, 31, 33, 35]
            labels = ['25-27', '27-29', '29-31', '31-33', '33-35']
        
        # Create binned column
        binned_col_name = f"{selected_num_col}_Binned"
        df[binned_col_name] = pd.cut(df[selected_num_col], bins=bins, labels=labels, right=False)
        
        # Create frequency table
        freq_table = create_frequency_table(df, binned_col_name)
        st.table(freq_table)
        
        st.markdown(download_csv(freq_table, f"frequency_{binned_col_name}"), unsafe_allow_html=True)
        
        # Show distribution
        fig = px.histogram(df, x=selected_num_col, nbins=10,
                        title=f"Distribution of {selected_num_col}")
        st.plotly_chart(fig, use_container_width=True)

# Cross-Tabulations
elif page == "Cross-Tabulations":
    st.title("Cross-Tabulation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        row_var = st.selectbox("Select Row Variable", 
                            ['Gender', 'Grade_Level', 'Extracurricular_Participation',
                            'Noise_Level_1to10_Category', 'Stress_Level_1to10_Category'])
    
    with col2:
        # Determine appropriate column variables based on row selection
        if row_var in ['Gender', 'Grade_Level']:
            default_col_vars = ['Overall_School_Satisfaction_1to10_Category', 
                            'Stress_Level_1to10_Category', 'Academic_Performance_Grade']
        else:
            default_col_vars = ['Overall_School_Satisfaction_1to10_Category', 
                            'Academic_Performance_Grade', 'Perceived_Safety_1to10_Category']
            
        col_var = st.selectbox("Select Column Variable", default_col_vars)
    
    # Create crosstab
    if 'Category' in col_var:
        cross_tab = pd.crosstab(df[row_var], df[col_var], normalize='index')*100
        title = f"% of {row_var} in each {col_var}"
        chart_type = "heatmap"
    else:
        # For numerical column variables, calculate the mean
        cross_tab = df.groupby(row_var)[col_var].mean().reset_index()
        title = f"Average {col_var} by {row_var}"
        chart_type = "bar"
    
    # Display the crosstab
    st.subheader(f"Cross-Tabulation: {row_var} vs {col_var}")
    st.table(cross_tab)
    
    st.markdown(download_csv(cross_tab.reset_index(), f"crosstab_{row_var}_vs_{col_var}"), 
            unsafe_allow_html=True)
    
    # Visualize the crosstab
    if chart_type == "heatmap":
        fig = px.imshow(cross_tab, text_auto='.1f', aspect="auto",
                    title=title,
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    else:  # bar chart
        fig = px.bar(cross_tab, x=row_var, y=col_var,
                title=title,
                color=row_var)
        st.plotly_chart(fig, use_container_width=True)
    
    # Chi-square test for categorical variables
    if chart_type == "heatmap":
        from scipy.stats import chi2_contingency
        
        # Create contingency table for chi-square
        contingency = pd.crosstab(df[row_var], df[col_var])
        
        # Calculate chi-square
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        st.subheader("Statistical Significance Test")
        st.write(f"Chi-square value: {chi2:.2f}")
        st.write(f"p-value: {p:.4f}")
        
        if p < 0.05:
            st.success(f"There is a statistically significant relationship between {row_var} and {col_var} (p < 0.05)")
        else:
            st.info(f"No statistically significant relationship detected between {row_var} and {col_var} (p > 0.05)")

# Advanced Visualizations
elif page == "Advanced Visualizations":
    st.title("Advanced Data Visualizations")
    
    viz_type = st.selectbox("Select Visualization Type", 
                        ["Correlation Heatmap", "Bubble Chart", "Parallel Categories", 
                        "Box Plots", "Scatter Matrix"])
    
    if viz_type == "Correlation Heatmap":
        st.subheader("Correlation Matrix of Numerical Variables")
        
        # Select only numerical columns for correlation
        num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and '1to10' in col]
        
        # Calculate correlation
        corr = df[num_cols].corr()
        
        # Plot heatmap
        fig = px.imshow(corr, text_auto='.2f',
                    title="Correlation Matrix of Comfort Factors",
                    color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights from Correlations")
        
        # Find the strongest positive and negative correlations
        corr_values = corr.unstack().sort_values(ascending=False)
        corr_values = corr_values[corr_values < 1.0]  # Remove self-correlations
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Strongest Positive Correlations:")
            pos_corr = corr_values.head(5)
            for idx, val in pos_corr.items():
                st.write(f"{idx[0]} & {idx[1]}: {val:.2f}")
        
        with col2:
            st.write("Strongest Negative Correlations:")
            neg_corr = corr_values.tail(5)
            for idx, val in neg_corr.items():
                st.write(f"{idx[0]} & {idx[1]}: {val:.2f}")
    
    elif viz_type == "Bubble Chart":
        st.subheader("Multivariate Relationships")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox("X-Axis", 
                            ["Peer_Relationship_Quality_1to10", "Sense_of_Belonging_1to10", 
                                "Teacher_Relationship_Quality_1to10"])
        
        with col2:
            y_var = st.selectbox("Y-Axis", 
                            ["Academic_Performance_Grade", "Overall_School_Satisfaction_1to10", 
                                "Stress_Level_1to10"])
        
        with col3:
            size_var = st.selectbox("Bubble Size", 
                                ["Weekly_Study_Hours", "Extracurricular_Satisfaction_1to10"])
            
        color_var = st.selectbox("Color By", 
                            ["Gender", "Grade_Level", "Extracurricular_Participation"])
        
        # Create bubble chart
        fig = px.scatter(df, x=x_var, y=y_var, size=size_var, color=color_var,
                    hover_name="Student_ID", size_max=20,
                    title=f"Relationship between {x_var}, {y_var}, {size_var} by {color_var}")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add trend line
        show_trend = st.checkbox("Show Trend Line")
        
        if show_trend:
            fig = px.scatter(df, x=x_var, y=y_var, size=size_var, color=color_var,
                        hover_name="Student_ID", size_max=20, trendline="ols",
                        title=f"Relationship with Trend Line")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display trend line equation
            import statsmodels.api as sm
            X = sm.add_constant(df[x_var])
            model = sm.OLS(df[y_var], X).fit()
            st.write(f"Regression Equation: {y_var} = {model.params[1]:.2f} × {x_var} + {model.params[0]:.2f}")
            st.write(f"R-squared: {model.rsquared:.3f}")
    
    elif viz_type == "Parallel Categories":
        st.subheader("Parallel Categories Plot")
        
        # Select categorical variables
        selected_cats = st.multiselect("Select Categories to Include", 
                                    ['Gender', 'Grade_Level', 'Extracurricular_Participation',
                                    'Noise_Level_1to10_Category', 'Stress_Level_1to10_Category',
                                    'Overall_School_Satisfaction_1to10_Category'],
                                    default=['Gender', 'Grade_Level', 'Overall_School_Satisfaction_1to10_Category'])
        
        if not selected_cats:
            st.warning("Please select at least one category")
        else:
            color_var = st.selectbox("Color By", selected_cats)
            
            # Create parallel categories plot
            fig = px.parallel_categories(df, dimensions=selected_cats, color=color_var,
                                    title="Parallel Categories Analysis")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plots":
        st.subheader("Box Plot Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            y_var = st.selectbox("Select Variable for Box Plot", 
                            ["Overall_School_Satisfaction_1to10", "Stress_Level_1to10", 
                                "Academic_Performance_Grade", "Perceived_Safety_1to10"])
        
        with col2:
            x_var = st.selectbox("Group By", 
                            ["Gender", "Grade_Level", "Extracurricular_Participation"])
        
        # Create box plot
        fig = px.box(df, x=x_var, y=y_var, color=x_var,
                title=f"Distribution of {y_var} by {x_var}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display statistics
        st.subheader("Summary Statistics")
        stats = df.groupby(x_var)[y_var].describe()
        st.table(stats)
    
    elif viz_type == "Scatter Matrix":
        st.subheader("Scatter Plot Matrix")
        
        # Select variables for scatter matrix
        vars_options = ["Overall_School_Satisfaction_1to10", "Stress_Level_1to10", 
                    "Academic_Performance_Grade", "Teacher_Relationship_Quality_1to10",
                    "Peer_Relationship_Quality_1to10", "Sense_of_Belonging_1to10",
                    "Perceived_Safety_1to10"]
        
        selected_vars = st.multiselect("Select Variables (3-4 recommended)", 
                                    vars_options,
                                    default=vars_options[:3])
        
        color_var = st.selectbox("Color By", 
                            ["Gender", "Grade_Level", "Overall_School_Satisfaction_1to10_Category"])
        
        if len(selected_vars) < 2:
            st.warning("Please select at least 2 variables")
        else:
            # Create scatter matrix
            fig = px.scatter_matrix(df, dimensions=selected_vars, color=color_var,
                                title="Scatter Plot Matrix")
            st.plotly_chart(fig, use_container_width=True)

# Predictive Analytics
elif page == "Predictive Analytics":
    st.title("Predictive Analytics")
    
    prediction_type = st.selectbox("Select Prediction Type", 
                                ["Overall Satisfaction Prediction", "Academic Performance Prediction",
                                "Stress Level Prediction", "Custom Prediction"])
    
    # Prepare data for prediction
    # First, encode categorical variables
    categorical_cols = ['Gender', 'Extracurricular_Participation', 
                    'Library_Satisfaction', 'Cafeteria_Satisfaction', 
                    'Bathroom_Cleanliness_Satisfaction', 'Computer_Lab_Access_Satisfaction',
                    'Internet_Access_Frequency']
    
    # Create a copy of the dataframe for model training
    model_df = df.copy()
    
    # Apply label encoding to categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col])
        label_encoders[col] = le
    
    # Define target variables based on prediction type
    if prediction_type == "Overall Satisfaction Prediction":
        target_var = "Overall_School_Satisfaction_1to10"
        exclude_cols = ["Student_ID", "Survey_Date", target_var]
        features = [col for col in model_df.columns if col not in exclude_cols and '_Category' not in col]
    
    elif prediction_type == "Academic Performance Prediction":
        target_var = "Academic_Performance_Grade"
        exclude_cols = ["Student_ID", "Survey_Date", target_var]
        features = [col for col in model_df.columns if col not in exclude_cols and '_Category' not in col]
    
    elif prediction_type == "Stress Level Prediction":
        target_var = "Stress_Level_1to10"
        exclude_cols = ["Student_ID", "Survey_Date", target_var]
        features = [col for col in model_df.columns if col not in exclude_cols and '_Category' not in col]
    
    else:  # Custom Prediction
        # Let user select target variable
        target_var = st.selectbox("Select Target Variable to Predict", 
                                [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                                and col != "Student_ID"])
        
        # Let user select features
        all_possible_features = [col for col in model_df.columns 
                            if col != target_var and col != "Student_ID" 
                            and col != "Survey_Date" and '_Category' not in col]
        
        features = st.multiselect("Select Features for Prediction", 
                                all_possible_features,
                                default=all_possible_features[:5])
    
    # Train model
    if st.button("Train Prediction Model"):
        with st.spinner("Training model..."):
            # Prepare features and target
            X = model_df[features]
            y = model_df[target_var]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            r2 = model.score(X_test, y_test)
            
            # Display results
            st.success("Model trained successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Squared Error", f"{mse:.2f}")
            
            with col2:
                st.metric("Root Mean Squared Error", f"{rmse:.2f}")
            
            with col3:
                st.metric("R² Score", f"{r2:.2f}")
            
            # Feature importance
            feature_imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
            feature_imp = feature_imp.sort_values('Importance', ascending=False)
            
            st.subheader("Feature Importance")
            fig = px.bar(feature_imp.head(10), x='Importance', y='Feature', orientation='h',
                    title=f"Top 10 Features for Predicting {target_var}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot actual vs predicted
            results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            
            st.subheader("Actual vs Predicted Values")
            fig = px.scatter(results, x='Actual', y='Predicted', 
                        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                        title=f"Actual vs Predicted {target_var}")
            
            # Add perfect prediction line
            min_val = min(results['Actual'].min(), results['Predicted'].min())
            max_val = max(results['Actual'].max(), results['Predicted'].max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                mode='lines', name='Perfect Prediction', 
                                line=dict(color='red', dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interactive prediction tool
            st.subheader("Make Custom Predictions")
            st.write("Adjust the values below to predict the outcome:")
            
            # Create input widgets for each feature
            input_values = {}
            
            for feature in features:
                if feature in categorical_cols:
                    # For categorical features, use the original categories
                    original_categories = df[feature].unique().tolist()
                    input_values[feature] = st.selectbox(f"Select {feature}", original_categories)
                    # Convert to encoded value
                    input_values[feature] = label_encoders[feature].transform([input_values[feature]])[0]
                else:
                    # For numerical features, use a slider
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    step = (max_val - min_val) / 100
                    
                    if feature.endswith('1to10'):
                        input_values[feature] = st.slider(f"Select {feature}", 1, 10, 5)
                    else:
                        default_val = float(df[feature].mean())
                        input_values[feature] = st.slider(f"Select {feature}", 
                                                    min_val, max_val, default_val, step)
            
            # Make prediction
            if st.button("Predict"):
                # Create input dataframe
                input_df = pd.DataFrame([input_values])
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Display prediction
                st.success(f"Predicted {target_var}: {prediction:.2f}")
                
                # Get similar students
                from sklearn.neighbors import NearestNeighbors
                
                nn = NearestNeighbors(n_neighbors=3)
                nn.fit(X)
                
                distances, indices = nn.kneighbors(input_df)
                
                st.subheader("Similar Students")
                st.write("These students have similar characteristics to your input:")
                
                similar_students = df.iloc[indices[0]]
                st.dataframe(similar_students[['Student_ID', target_var] + features])

# Cluster Analysis
elif page == "Cluster Analysis":
    st.title("Student Cluster Analysis")
    
    # Select variables for clustering
    st.subheader("Select Variables for Clustering")
    
    clustering_vars = st.multiselect(
        "Select Variables",
        options=[col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
            and col != "Student_ID" and '_Category' not in col],
        default=["Overall_School_Satisfaction_1to10", "Academic_Performance_Grade", 
                "Stress_Level_1to10", "Sense_of_Belonging_1to10"]
    )
    
    if not clustering_vars:
        st.warning("Please select at least two variables for clustering")
    elif len(clustering_vars) < 2:
        st.warning("Please select at least two variables for clustering")
    else:
        # Number of clusters
        num_clusters = st.slider("Select Number of Clusters", 2, 6, 3)
        
        if st.button("Perform Cluster Analysis"):
            with st.spinner("Clustering students..."):
                # Prepare data for clustering
                X = df[clustering_vars].copy()
                
                # Standardize the data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply K-means clustering
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Add cluster labels to dataframe
                df_clustered = df.copy()
                df_clustered['Cluster'] = clusters
                
                # Display results
                st.success(f"Students grouped into {num_clusters} clusters successfully!")
                
                # Cluster distribution
                st.subheader("Cluster Distribution")
                cluster_counts = df_clustered['Cluster'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                
                fig = px.pie(cluster_counts, values='Count', names='Cluster', 
                        title="Distribution of Students Across Clusters")
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster profiles
                st.subheader("Cluster Profiles")
                
                # Calculate mean values for each cluster
                cluster_profiles = df_clustered.groupby('Cluster')[clustering_vars].mean()
                
                # Display cluster profiles
                st.table(cluster_profiles)
                
                # Radar chart for cluster profiles
                st.subheader("Cluster Comparison")
                
                # Normalize the values for radar chart
                normalized_profiles = cluster_profiles.copy()
                for col in normalized_profiles.columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    normalized_profiles[col] = (normalized_profiles[col] - min_val) / (max_val - min_val)
                
                # Create radar chart
                fig = go.Figure()
                
                for cluster in range(num_clusters):
                    fig.add_trace(go.Scatterpolar(
                        r=normalized_profiles.iloc[cluster].values,
                        theta=normalized_profiles.columns,
                        fill='toself',
                        name=f'Cluster {cluster}'
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Normalized Cluster Profiles"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Visualize clusters in 2D
                st.subheader("Cluster Visualization")
                
                if len(clustering_vars) >= 3:
                    # Use 3D scatter plot for visualization
                    x_var = clustering_vars[0]
                    y_var = clustering_vars[1]
                    z_var = clustering_vars[2]
                    
                    fig = px.scatter_3d(df_clustered, x=x_var, y=y_var, z=z_var,
                                    color='Cluster', hover_name="Student_ID",
                                    title=f"3D Visualization of Clusters")
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Use 2D scatter plot
                    x_var = clustering_vars[0]
                    y_var = clustering_vars[1]
                    
                    fig = px.scatter(df_clustered, x=x_var, y=y_var,
                                color='Cluster', hover_name="Student_ID",
                                title=f"2D Visualization of Clusters")
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster descriptions
                st.subheader("Cluster Descriptions")
                
                # Generate descriptive names for clusters
                for cluster in range(num_clusters):
                    profile = cluster_profiles.iloc[cluster]
                    
                    # Find the highest and lowest values
                    highest_val = profile.idxmax()
                    lowest_val = profile.idxmin()
                    
                    overall_satisfaction = profile['Overall_School_Satisfaction_1to10'] if 'Overall_School_Satisfaction_1to10' in profile else None
                    
                    # Generate description
                    description = f"Cluster {cluster} ({len(df_clustered[df_clustered['Cluster'] == cluster])} students): "
                    
                    if overall_satisfaction is not None:
                        if overall_satisfaction >= 8:
                            description += "High overall satisfaction. "
                        elif overall_satisfaction >= 5:
                            description += "Moderate overall satisfaction. "
                        else:
                            description += "Low overall satisfaction. "
                    
                    description += f"Strongest in {highest_val}, weakest in {lowest_val}."
                    
                    st.write(description)
                
                # Download cluster assignments
                cluster_results = df_clustered[['Student_ID', 'Cluster'] + clustering_vars]
                st.markdown(download_csv(cluster_results, "student_clusters"), unsafe_allow_html=True)

# Recommendations
elif page == "Recommendations":
    st.title("Recommendations for Improvement")
    
    # Overall satisfaction analysis
    overall_satisfaction = df['Overall_School_Satisfaction_1to10'].mean()
    
    st.subheader("Overall School Satisfaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = overall_satisfaction,
            title = {'text': "Average Overall Satisfaction"},
            gauge = {'axis': {'range': [0, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 3], 'color': "red"},
                    {'range': [3, 7], 'color': "yellow"},
                    {'range': [7, 10], 'color': "green"}
                ]}
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution of overall satisfaction
        fig = px.histogram(df, x='Overall_School_Satisfaction_1to10',
                        title="Distribution of Overall Satisfaction",
                        color_discrete_sequence=['blue'])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    # Identify areas for improvement
    st.subheader("Priority Areas for Improvement")
    
    # Calculate the mean of each rating variable
    rating_cols = [col for col in df.columns if '1to10' in col and 'Category' not in col 
                and col != 'Overall_School_Satisfaction_1to10']
    
    ratings = df[rating_cols].mean().reset_index()
    ratings.columns = ['Factor', 'Average Rating']
    ratings = ratings.sort_values('Average Rating')
    
    # Identify bottom 5 factors
    bottom_factors = ratings.head(5)
    
    fig = px.bar(bottom_factors, x='Average Rating', y='Factor', orientation='h',
            title="Lowest Rated Comfort Factors",
            color='Average Rating',
            color_continuous_scale='Reds_r')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation with overall satisfaction
    st.subheader("Impact on Overall Satisfaction")
    
    # Calculate correlation with overall satisfaction
    correlations = {}
    for col in rating_cols:
        corr = df[col].corr(df['Overall_School_Satisfaction_1to10'])
        correlations[col] = corr
    
    corr_df = pd.DataFrame(list(correlations.items()), columns=['Factor', 'Correlation'])
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    # Top 5 factors with highest correlation
    top_corr_factors = corr_df.head(5)
    
    fig = px.bar(top_corr_factors, x='Correlation', y='Factor', orientation='h',
            title="Factors with Highest Impact on Overall Satisfaction",
            color='Correlation',
            color_continuous_scale='Greens')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate recommendations
    st.subheader("Automated Recommendations")
    
    # Find factors that are both low-rated and highly correlated with satisfaction
    priority_factors = pd.merge(bottom_factors, corr_df, on='Factor')
    priority_factors['Priority Score'] = priority_factors['Correlation'] / priority_factors['Average Rating']
    priority_factors = priority_factors.sort_values('Priority Score', ascending=False)
    
    # Display priority matrix
    st.write("Priority Matrix (Lower Rating + Higher Impact = Higher Priority)")
    st.table(priority_factors)
    
    # Generate specific recommendations
    st.subheader("Specific Recommendations")
    
    for i, row in priority_factors.iterrows():
        factor = row['Factor']
        rating = row['Average Rating']
        impact = row['Correlation']
        
        st.write(f"**Focus Area: {factor.replace('_1to10', '').replace('_', ' ')}**")
        
        # Generate recommendation based on factor
        if 'Noise_Level' in factor:
            st.write("- Implement noise reduction strategies in classrooms")
            st.write("- Consider sound-absorbing materials for walls and ceilings")
            st.write("- Establish clear noise level expectations for different activities")
        
        elif 'Cleanliness' in factor:
            st.write("- Increase frequency of cleaning schedules")
            st.write("- Implement student-led cleanliness initiatives")
            st.write("- Improve waste management systems in common areas")
        
        elif 'Seating_Comfort' in factor:
            st.write("- Evaluate current seating and consider ergonomic alternatives")
            st.write("- Allow students brief standing or stretching breaks during long classes")
            st.write("- Consider flexible seating options for different learning activities")
        
        elif 'Stress_Level' in factor:
            st.write("- Implement stress management workshops for students")
            st.write("- Review homework policies and assignment scheduling")
            st.write("- Create quiet spaces for relaxation during breaks")
        
        elif 'Belonging' in factor:
            st.write("- Increase opportunities for student collaboration and community building")
            st.write("- Create more inclusive school events and activities")
            st.write("- Implement peer mentoring programs for new students")
        
        elif 'Safety' in factor:
            st.write("- Review and strengthen school safety protocols")
            st.write("- Increase adult presence in common areas")
            st.write("- Create clear reporting mechanisms for safety concerns")
        
        elif 'Teacher_Relationship' in factor:
            st.write("- Provide professional development focused on student-teacher relationships")
            st.write("- Create more opportunities for positive student-teacher interactions")
            st.write("- Implement regular check-ins between teachers and students")
        
        elif 'Peer_Relationship' in factor:
            st.write("- Facilitate structured social activities to promote friendship")
            st.write("- Implement conflict resolution training")
            st.write("- Create more collaborative learning opportunities")
        
        else:
            st.write("- Review current policies and resources in this area")
            st.write("- Survey students for specific improvement suggestions")
            st.write("- Implement targeted interventions based on feedback")
        
        st.write("---")
    
    # Implementation timeline
    st.subheader("Recommended Implementation Timeline")
    
    timeline_data = {
        'Task': [f"Address {factor.replace('_1to10', '').replace('_', ' ')}" for factor in priority_factors['Factor']],
        'Start': [f"Month {i+1}" for i in range(len(priority_factors))],
        'Duration': [np.random.randint(1, 4) for _ in range(len(priority_factors))]
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    timeline_df['Finish'] = [f"Month {int(timeline_df['Start'][i].split()[1]) + timeline_df['Duration'][i]}" 
                        for i in range(len(timeline_df))]
    
    st.table(timeline_df)
    
    # Success metrics
    st.subheader("Recommended Success Metrics")
    
    st.write("1. **Improvement in Targeted Areas**")
    st.write("   - Measure: Pre/post intervention surveys")
    st.write("   - Target: 20% improvement in ratings for priority factors")
    
    st.write("2. **Overall Satisfaction Increase**")
    st.write("   - Measure: Average overall satisfaction score")
    st.write("   - Target: 15% increase within one academic year")
    
    st.write("3. **Academic Performance Correlation**")
    st.write("   - Measure: Correlation between comfort improvements and academic grades")
    st.write("   - Target: Positive correlation coefficient of 0.3 or higher")

# Raw Data
elif page == "Raw Data":
    st.title("Raw Survey Data")
    
    # Filter options
    st.sidebar.header("Filter Data")
    
    # Gender filter
    gender_filter = st.sidebar.multiselect(
        "Select Gender",
        options=df['Gender'].unique(),
        default=df['Gender'].unique()
    )
    
    # Grade level filter
    grade_filter = st.sidebar.multiselect(
        "Select Grade Level",
        options=df['Grade_Level'].unique(),
        default=df['Grade_Level'].unique()
    )
    
    # Apply filters
    filtered_df = df[df['Gender'].isin(gender_filter) & df['Grade_Level'].isin(grade_filter)]
    
    # Show data
    st.dataframe(filtered_df)
    
    # Download options
    st.subheader("Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(download_csv(filtered_df, "filtered_student_data"), unsafe_allow_html=True)
    
    with col2:
        if st.button("Generate Sample Data"):
            st.markdown(download_csv(generate_sample_data(), "sample_student_data"), unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("### Enhancing Youth Experiences: Student Comfortability Analysis Tool")
st.markdown("Developed for Davao City National High School")