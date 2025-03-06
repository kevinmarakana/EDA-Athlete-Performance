# EDA-Athlete-Performance
# %% [markdown]
# # **Exploratory Data Analysis on the Athlete Performance Dataset**
# 
# ###**Perform EDA BY :** Kevin Marakana
# 
# ###**Objective:**
# The primary objective of this analysis is to explore the Athlete Performance Dataset to
# identify key insights and trends that influence athletic performance. Additionally, we
# aim to perform exploratory data analysis on athletes' physical, training, and nutrition
# parameters. This project can help sports scientists, coaches, and data analysts
# optimize training programs and improve athletic outcomes.
# 
# ###**Dataset Information:**
# The dataset contains detailed information about athletes across various sports,
# focusing on their physical condition, training, nutrition, and historical performance
# metrics. The data is structured to assist in exploratory data analysis (EDA).
# 
# 
# *   **Source:** https://www.datayb.com/datasets/dataset-details/datayb_dataset_details_wqpn8mq4mubs5l8/
# 
# 
# 
# ###**Athlete Performance Dataset: Column Information**
# 
# | Column Name | Description | Data Type | Possible Values |
# |-------------|-------------|-----------|-----------------|
# | Athlete_ID | Unique identifier for each athlete. | String | Unique alphanumeric IDs (e.g., "A001", "A002"). |
# | Athlete_Name | Name of the athlete. | String | Varies by athlete. |
# | Sport_Type | Type of sport the athlete participates in. | String | Running, Swimming, Cycling, Basketball, Soccer, Tennis, etc. |
# | Event | Specific event the athlete is competing in. | String | 100m Sprint, Marathon, 50m Freestyle, 10km Road Race, etc. |
# | Training_Hours_per_Week | Weekly training hours of the athlete. | Float | 0.0 to 40.0 (hours). |
# | Average_Heart_Rate | Average heart rate during training. | Float | 50.0 to 200.0 (beats per minute). |
# | BMI | Body Mass Index of the athlete. | Float | 15.0 to 35.0. |
# | Sleep_Hours_per_Night | Average sleep hours per night. | Float | 4.0 to 10.0 (hours). |
# | Daily_Caloric_Intake | Daily calorie intake of the athlete. | Float | 1500.0 to 4000.0 (calories). |
# | Hydration_Level | Hydration percentage of the athlete. | Float | 50.0 to 100.0 (%). |
# | Injury_History | Past injury details. | String | None, Minor, Major. |
# | Previous_Competition_Performance | Performance score or time in previous competitions. | Float | 0.0 to 100.0 (score or time). |
# | Training_Intensity | Intensity level of training. | String | Low, Medium, High. |
# | Resting_Heart_Rate | Resting heart rate of the athlete. | Float | 40.0 to 100.0 (beats per minute). |
# | Body_Fat_Percentage | Percentage of body fat. | Float | 5.0 to 30.0 (%). |
# | VO2_Max | Maximum oxygen uptake during exercise. | Float | 30.0 to 80.0 (ml/kg/min). |
# | Event_Distance | Distance of the event the athlete is competing in. | Float | 50.0 to 5000.0 (meters). |
# | Altitude_Training | Whether altitude training is part of the regime. | String | None, Low, Medium, High. |
# | Mental_Focus_Level | Level of mental focus. | Float | 1.0 to 10.0 (scale). |
# | Competition_Date | Date of the competition. | Datetime/String | Various dates. |
# | Performance_Metric | Target variable measuring athlete performance. | Float | Time, Score, Distance (depending on sport/event). |
# 
# 

# %% [markdown]
# ###**Steps to Achieve the Task:**
# 1.   **Data Understanding:**
# 
#   *   Load the dataset and examine the structure and summary statistics.
#   *   Check for null or missing values and handle them appropriately.
#   *   Analyze data types for consistency.
# 
# 2.  **Descriptive Statistics:**
# 
#   *   Generate summary statistics for numerical and categorical columns.
#   *   Calculate measures of central tendency (mean, median) and dispersion
# (variance, standard deviation).
# 
# 3. **Data Visualization:**
#   *   Create visualizations (histograms, bar charts, scatter plots) to explore
# each column.
#   *   Use correlation heatmaps to identify relationships between numerical
# variables.
# 
# 4. **Data Cleaning:**
#   *   Remove or impute missing values.
#   *   Normalize or standardize numerical data if required.
# 
# 5. **Exploratory Analysis:**
# 
#   * Analyze patterns, trends, and categorical data distributions.
#   * Identify correlations between features and the target variable
# (Performance_Metric).
# 
# 6. **Insight Generation:**
# 
#   * Derive at least 25 meaningful insights from the data.
# 
#   * Support insights with graphs and visualizations.
# 
# 
# 
# 
# 
# 
# 
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import io
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv('/content/drive/MyDrive/EDA/sports_performance_data.csv')

# %%
print("****************************")
print(" ")
print(f"Dataset Shape: {df.shape}")
print(" ")
print("****************************")
print(" ")
display(df.head())

# %%
print("**** Data Type Information ****")
display(df.dtypes)


# %%
print(" ******** Missing Value Information ********")
print(" ")
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})
display(missing_df[missing_df['Missing Values'] > 0])

# %%
print(" ******* Descriptive Statistics *******")
print(" ")
display(df.describe())


# %%
df.head()

# %%
print(" ******* Handaling Missing Value ******* ")
print(" ")
# Fill numerical columns with median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"Filled {col} missing values with median: {median_value}")

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"Filled {col} missing values with mode: {mode_value}")

# %%
# Check if missing values are resolved
df.isnull().sum()

# %%
print("\n--- Sport Distribution Analysis ---")
print(" ")
plt.figure(figsize=(12, 6))
sport_counts = df['Sport_Type'].value_counts()
sns.barplot(x=sport_counts.index, y=sport_counts.values)
plt.title('Distribution of Athletes by Sport Type')
plt.xlabel('Sport Type')
plt.ylabel('Number of Athletes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
print("\n--- Event Distribution Analysis ---")
print(" ")
plt.figure(figsize=(14, 8))
event_counts = df['Event'].value_counts().head(15)  # Top 15 events
sns.barplot(x=event_counts.index, y=event_counts.values)
plt.title('Top 15 Events by Number of Athletes')
plt.xlabel('Event')
plt.ylabel('Number of Athletes')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# # **Insights and Suggested Graphs**
# 
# 1. **Insight**: Athletes with higher training hours per week generally show better performance metrics.  
#    **Suggested Graph**: Scatter plot of Training_Hours_per_Week vs. Performance_Metric.

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Training_Hours_per_Week', y='Performance_Metric', data=df,
                hue='Sport_Type', size='Training_Intensity', sizes=(50, 200), alpha=0.7)
plt.title('Training Hours per Week vs Performance Metric')
plt.xlabel('Training Hours per Week')
plt.ylabel('Performance Metric')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 2. **Insight**:  Lower BMI values correlate with better performance in endurance
# sports like running and cycling.  
#    **Suggested Graph**: Boxplot of BMI grouped by Sport_Type.

# %% [markdown]
# .

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(x='Sport_Type', y='BMI', data=df)
plt.title('BMI Distribution by Sport Type')
plt.xlabel('Sport Type')
plt.ylabel('BMI')
plt.xticks(rotation=45)
plt.axhline(y=25, color='r', linestyle='--', label='Overweight Threshold')
plt.axhline(y=18.5, color='g', linestyle='--', label='Underweight Threshold')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 3. **Insight**:   Sleep hours per night significantly impact performance, with optimal
# range being 7-9 hours.  
#    **Suggested Graph**: Line graph of Sleep_Hours_per_Night vs.
# Performance_Metric.

# %% [markdown]
# .

# %%
plt.figure(figsize=(10, 6))
sns.lineplot(x='Sleep_Hours_per_Night', y='Performance_Metric', data=df, ci=None)
plt.title('Sleep Hours per Night vs Performance Metric')
plt.xlabel('Sleep Hours per Night')
plt.ylabel('Performance Metric')
plt.axvline(x=7, color='g', linestyle='--', label='Min. Recommended Sleep')
plt.axvline(x=9, color='r', linestyle='--', label='Max. Recommended Sleep')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 4. **Insight**:   Higher hydration levels result in improved performance across all
# sports.
# 
#   **Suggested Graph**: Scatter plot of Hydration_Level vs. Performance_Metric.

# %% [markdown]
# .

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Hydration_Level', y='Performance_Metric', data=df, hue='Sport_Type', alpha=0.7)
plt.title('Hydration Level vs Performance Metric')
plt.xlabel('Hydration Level (%)')
plt.ylabel('Performance Metric')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 5. **Insight**:    Athletes with consistent high mental focus tend to perform better in
# high-pressure events.
# 
#    **Suggested Graph**:  Bar chart of Mental_Focus_Level vs. Performance_Metric
# grouped by Event.

# %% [markdown]
# .

# %%

sns.barplot(x='Mental_Focus_Level', y='Performance_Metric', data=df, hue='Event')
plt.title('Mental Focus Level vs Performance Metric by Event')
plt.xlabel('Mental Focus Level')
plt.ylabel('Performance Metric')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 6. **Insight**:     Altitude training is more common among endurance athletes and
# positively impacts VO2_Max.
# 
#    **Suggested Graph**:  Pie chart of Altitude_Training distribution by Sport_Type.

# %% [markdown]
# .

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

altitude_by_sport = pd.crosstab(df['Altitude_Training'], df['Sport_Type'])
labels = altitude_by_sport.columns

for i, (altitude, data) in enumerate(altitude_by_sport.iterrows()):
    data.plot.pie(
        ax=axs[i],
        autopct='%1.1f%%',
        fontsize=16,
        labels=altitude_by_sport.columns
    )
    axs[i].set_title(f'{altitude}', fontsize=18)


fig.suptitle('Sport Type Distribution by Altitude Training', fontsize=16, y=0.98)
fig.legend(labels, bbox_to_anchor=(1.08, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 7. **Insight**:      Body fat percentage is inversely proportional to performance in most
# sports.  
# 
#    **Suggested Graph**:   Line graph of Body_Fat_Percentage vs.
# Performance_Metric.

# %% [markdown]
# .

# %%
plt.figure(figsize=(10, 6))
sns.lineplot(x='Body_Fat_Percentage', y='Performance_Metric', data=df)
plt.title('Body Fat Percentage vs Performance Metric')
plt.xlabel('Body Fat Percentage (%)')
plt.ylabel('Performance Metric')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 8. **Insight**:      Injury history significantly affects performance; athletes with major
# injuries show reduced metrics.
# sports.  
# 
#    **Suggested Graph**:   Boxplot of Performance_Metric grouped by Injury_History.

# %% [markdown]
# .

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x='Injury_History', y='Performance_Metric', data=df)
plt.title('Performance Metric by Injury History')
plt.xlabel('Injury History')
plt.ylabel('Performance Metric')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 9. **Insight**:      Resting heart rate has a strong correlation with endurance event
# performance.  
# 
#    **Suggested Graph**:   Scatter plot of Resting_Heart_Rate vs.
# Performance_Metric for endurance sports.

# %% [markdown]
# .

# %%
endurance_sports = ['Running', 'Swimming', 'Cycling', 'Triathlon']
endurance_df = df[df['Sport_Type'].isin(endurance_sports)]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Resting_Heart_Rate', y='Performance_Metric',
                data=endurance_df, hue='Sport_Type', alpha=0.7)
plt.title('Resting Heart Rate vs Performance Metric for Endurance Sports')
plt.xlabel('Resting Heart Rate (bpm)')
plt.ylabel('Performance Metric')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# %% [markdown]
# .

# %% [markdown]
#     
# 10. **Insight**:      Higher daily caloric intake aligns with better performance in high
# intensity sports.   
# 
#    **Suggested Graph**:   Bar chart of Daily_Caloric_Intake vs. Performance_Metric
# grouped by Training_Intensity.

# %% [markdown]
# .

# %%
plt.figure(figsize=(10, 6))
sns.barplot(x='Training_Intensity', y='Performance_Metric',
            hue='Daily_Caloric_Intake', data=df)
plt.title('Performance Metric by Training Intensity and Daily Caloric Intake')
plt.xlabel('Training Intensity')
plt.ylabel('Performance Metric')
plt.legend(title='Daily Caloric Intake' ,bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 11. **Insight**:      Athletes in sports like swimming and cycling exhibit higher VO2_Max
# values compared to others.   
# 
#    **Suggested Graph**:   Boxplot of VO2_Max grouped by Sport_Type.
# 

# %% [markdown]
# .

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(x='Sport_Type', y='VO2_Max', data=df)
plt.title('VO2 Max Distribution by Sport Type')
plt.xlabel('Sport Type')
plt.ylabel('VO2 Max (ml/kg/min)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 12. **Insight**:       Performance scores are higher in athletes with medium to high
# training intensity.   
# 
#    **Suggested Graph**:    Bar chart of Training_Intensity vs. Performance_Metric.
# 

# %% [markdown]
# .

# %%
plt.figure(figsize=(10, 6))
sns.barplot(x='Training_Intensity', y='Performance_Metric', data=df)
plt.title('Training Intensity vs Performance Metric')
plt.xlabel('Training Intensity')
plt.ylabel('Performance Metric')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 13. **Insight**:       Previous competition performance is a strong predictor of future
# performance metrics.   
# 
#    **Suggested Graph**:   Line graph of Previous_Competition_Performance vs.
# Performance_Metric.

# %% [markdown]
# .

# %%
plt.figure(figsize=(10, 6))
sns.lineplot(x='Previous_Competition_Performance', y='Performance_Metric', data=df)
plt.title('Previous Competition Performance vs Current Performance Metric')
plt.xlabel('Previous Competition Performance')
plt.ylabel('Current Performance Metric')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 14. **Insight**:       Event distance impacts training regimes; longer distances correlate
# with higher training hours.
# 
#    **Suggested Graph**:     Scatter plot of Event_Distance vs.
# Training_Hours_per_Week.

# %% [markdown]
# .

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Event_Distance', y='Training_Hours_per_Week',
                data=df, hue='Sport_Type', alpha=0.7)
plt.title('Event Distance vs Training Hours per Week')
plt.xlabel('Event Distance (meters)')
plt.ylabel('Training Hours per Week')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 15. **Insight**:       Mental focus levels are higher in individual sports compared to team
# sports.    
# 
#    **Suggested Graph**:     Boxplot of Mental_Focus_Level grouped by Sport_Type.
# 

# %% [markdown]
# .

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(x='Sport_Type', y='Mental_Focus_Level', data=df)
plt.title('Mental Focus Level by Sport Type')
plt.xlabel('Sport Type')
plt.ylabel('Mental Focus Level (1-10)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 16. **Insight**:      Competition date proximity affects athlete readiness and
# performance.
# 
#    **Suggested Graph**:    Line graph of Days to Competition_Date vs.
# Performance_Metric.

# %% [markdown]
# .

# %%

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=df['Competition_Date'], y=df['Performance_Metric'])
    plt.title('Days to Competition vs Performance Metric')
    plt.xlabel('Days to Competition')
    plt.ylabel('Performance Metric')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 17. **Insight**:      Athletes with high VO2_Max and low Resting_Heart_Rate excel in
# endurance events.   
# 
#    **Suggested Graph**:   Scatter plot of VO2_Max vs. Resting_Heart_Rate for
# endurance athletes.
# 

# %% [markdown]
# .

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='VO2_Max', y='Resting_Heart_Rate',
                data=endurance_df, hue='Sport_Type',
                sizes=(20, 200), alpha=0.7)
plt.title('VO2 Max vs Resting Heart Rate for Endurance Athletes')
plt.xlabel('VO2 Max (ml/kg/min)')
plt.ylabel('Resting Heart Rate (bpm)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 18. **Insight**:       Injury-prone athletes often reduce training hours, impacting
# performance.   
# 
#    **Suggested Graph**:   Line graph of Training_Hours_per_Week vs.
# Injury_History.

# %% [markdown]
# .

# %%
plt.figure(figsize=(10, 6))
sns.lineplot(x='Training_Hours_per_Week', y='Injury_History', data=df)
plt.title('Training Hours per Week vs Injury History')
plt.xlabel('Training Hours per Week')
plt.ylabel('Injury History')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 19. **Insight**:      Soccer and basketball players show higher average caloric intake
# compared to runners.   
# 
#    **Suggested Graph**:   Bar chart of Daily_Caloric_Intake grouped by Sport_Type.
# 

# %% [markdown]
# .

# %%
plt.figure(figsize=(12, 6))
sns.barplot(x='Sport_Type', y='Daily_Caloric_Intake', data=df)
plt.title('Daily Caloric Intake by Sport Type')
plt.xlabel('Sport Type')
plt.ylabel('Daily Caloric Intake (calories)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 20. **Insight**:       Athletes performing at higher altitudes tend to have higher hydration
# levels.   
# 
#    **Suggested Graph**:    Scatter plot and stripplot of Altitude_Training vs. Hydration_Level.
# 

# %% [markdown]
# .

# %%
plt.figure(figsize=(10, 6))
sns.stripplot(x='Altitude_Training', y='Hydration_Level', data=df,
             jitter=True, alpha=0.7, hue='Sport_Type')
plt.title('Altitude Training vs Hydration Level')
plt.xlabel('Altitude Training')
plt.ylabel('Hydration Level (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Altitude_Training', y='Hydration_Level', data=df, hue='Sport_Type',
                 alpha=0.7)
plt.title('Altitude Training vs Hydration Level')
plt.xlabel('Altitude Training')
plt.ylabel('Hydration Level (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 21. **Insight**:      Consistent sleep patterns improve overall mental focus and
# competition outcomes.   
# 
#    **Suggested Graph**:    Line graph of Sleep_Hours_per_Night vs.
# Mental_Focus_Level.
# 

# %% [markdown]
# .

# %%
plt.figure(figsize=(10, 6))
sns.lineplot(x='Sleep_Hours_per_Night', y='Mental_Focus_Level', data=df)
plt.title('Sleep Hours per Night vs Mental Focus Level')
plt.xlabel('Sleep Hours per Night')
plt.ylabel('Mental Focus Level (1-10)')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 22. **Insight**:      Younger athletes tend to perform better in speed-oriented sports like
# sprints.    
# 
#    **Suggested Graph**:    Bar chart of Age (derived) vs. Performance_Metric for
# speed events.
# 

# %% [markdown]
# .

# %%
#derived age from Athlete_Name column , it contain some random digit in range of 1 to 100 after _
df["Age"] = df["Athlete_Name"].str.extract(r'_(\d+)$').astype(int)
df["Age"].unique()

# %%
df.loc[(df["Age"] < 10) | (df["Age"] > 101), "Age"] = None

speed_sports = ["100m Sprint", "200m Sprint", "400m Sprint"]
df_speed = df[df["Event"].isin(speed_sports)]

plt.figure(figsize=(16, 8))
sns.barplot(x="Age", y="Performance_Metric", data=df_speed, ci=None)
plt.xlabel("Age")
plt.ylabel("Performance Metric (Time/Score)")
plt.title("Performance vs Age for Speed-Oriented Sports")
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 23. **Insight**:     Performance metrics improve significantly after injury recovery if
# training intensity is progressively increased.   
# 
#    **Suggested Graph**:    Line graph of Training_Intensity vs. Performance_Metric
# post-injury.

# %% [markdown]
# .

# %%
injured_athletes = df[df['Injury_History'] != 'None']
plt.figure(figsize=(10, 6))
sns.lineplot(x='Training_Intensity', y='Performance_Metric',
             data=injured_athletes, hue='Injury_History')
plt.title('Training Intensity vs Performance Metric Post-Injury')
plt.xlabel('Training Intensity')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel('Performance Metric')
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 24. **Insight**:     Athletes with higher event distances tend to have longer training
# sessions per week.   
# 
#    **Suggested Graph**:   Scatter plot of Event_Distance vs.
# Training_Hours_per_Week.
# 

# %% [markdown]
# .

# %%
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Event_Distance", y="Training_Hours_per_Week", data=df , hue = 'Sport_Type')
plt.xlabel("Event Distance (meters)")
plt.ylabel("Training Hours per Week")
plt.title("Training Hours vs Event Distance")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %% [markdown]
# .

# %% [markdown]
#     
# 25. **Insight**:       Body fat percentage shows a stronger correlation with performance
# in sports requiring high agility.    
# 
#    **Suggested Graph**:   Boxplot of Body_Fat_Percentage grouped by Sport_Type
# for agility-based events.
# 
# 

# %% [markdown]
# .

# %%
agility_sports = ['Tennis', 'Basketball', 'Soccer', 'Volleyball', 'Badminton']
agility_df = df[df['Sport_Type'].isin(agility_sports)]

plt.figure(figsize=(12, 6))
sns.boxplot(x='Sport_Type', y='Body_Fat_Percentage', data=agility_df)
plt.title('Body Fat Percentage by Sport Type for Agility-Based Sports')
plt.xlabel('Sport Type')
plt.ylabel('Body Fat Percentage (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# .

# %% [markdown]
# ##**Heatmap**

# %% [markdown]
# .

# %%

correlation_matrix = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(16, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.tight_layout()
plt.show()


