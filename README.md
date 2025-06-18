<h1 style="color:#EE2737FF"><center><font size="+6">Heart Analysis Prediction</font></center></h1>

![](https://media.istockphoto.com/id/1295615803/vector/tiny-doctors-examining-heart-health-flat-vector-illustration.jpg?s=612x612&w=0&k=20&c=lbgCsYF0hocLsxUP3qr7amT5LgRYFtTNbs4aBah_Bz0=)

<h3 style="color:#EE2737FF">Introduction</h3>

Cardiovascular diseases are serious problems with the heart and blood vessels, and they are one of the main causes of death around the world. In **2019**, nearly **18 million** people died from these diseases, making up **32%** of global deaths, according to the [**American Heart Association**](https://www.ahajournals.org/doi/10.1161/CIR.0000000000000950?utm_campaign=ahajournals&utm_source=heart&utm_medium=link&utm_content=statshome) (World Health Organization, 2021). Most of these deaths (**85%**) were caused by heart attacks and strokes, and a large part of them (**38%**) affected people under **70**.

Because of these alarming numbers, it’s crucial to catch cardiovascular diseases early. This is where machine learning can really make a difference. By analyzing patterns in health data, machine learning models can help predict who might be at risk before serious issues arise.

In this project, we’ll go step by step through the process of analyzing data and building a model that predicts the risk of cardiovascular disease. We’ll use a machine learnings model, and look at which factors are the most important for this prediction.

Mainly, the notebook aims to show the project in **6 parts**
1. [**Part 1**: Introduction (*Data Collection Phase*)](#001)
    - About Us
    - About Dataset
2. [**Part 2**: Data Loading (*Data Preparation & Wrangling Phase*)](#002)
    - Data Loading & Wrangling
    - Separating FLoat, Integer, & Categorical Columns
3. [**Part 3**: Exploratory Data Analysis (EDA) (*Data Analysis Phase*)](#003)
    - Ploting Basic Visualization
        - Plot Float Columns
        - Plot Integer Columns
        - Plot Categorical Columns
    - Plot Advanced Visualization
        - Plot Stats
        - Plot Distributions
        - Plot Correlations   
4. [**Part 4**: Machine Learning Modeling](#004)
    - Splting Data into Train, Validation, & Test Dataset
    - Logistic Regression
    - Random Forest Classifier
    - Gradient Boosting Classifier
5. [**Part 5**: Model Evaluation & Interpretation](#005)
    - Logistic Regression Results
    - Random Forest Classifier Results
    - Gradient Boosting Classifier Results
6. [**Part 6**: Conclusion](#006)
    - Conclusion
    - References


## <div style="color:rgb(0, 103, 71);display:fill;border-radius:5px;background-color:whie;letter-spacing:0.1px;overflow:hidden"><p style="padding:10px;color:rgb(0, 103, 71);overflow:hidden;margin:0;font-size:100%; text-align:center"><b id= '001'>Part 1 :</b> Data Collection</p></div>

<h3 id='01', style="color:#EE2737FF">About Dataset</h3>

In this study, we’re using [**Heart Disease Data**](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) that combines information from the UCI Machine Learning Repository. The data set includes **11** different **features** that can help predict the likelihood of heart failure—a serious heart condition that significantly raises the risk of death from cardiovascular issues. The key outcome we’re looking at is whether heart failure is present, shown by a simple indicator: if `HeartDisease = 1`, it means heart failure is diagnosed.

1. **Age**: The age of the patient, in years.
2. **Sex**: The patient's gender, male or female.
3. **ChestPainType**: The type of chest pain experienced by the patient:
    - **TA**: Typical Angina
    - **ATA**: Atypical Angina
    - **NAP**: Non-Anginal Pain
    - **ASY**: Asymptomatic
    
4. **RestingBP**: The patient's resting blood pressure (mmHg).
5. **Cholesterol**: The patient's serum cholesterol (mg/dl).
6. **FastingBS**: The patient's fasting blood sugar.
    - **1** if glucose > 120 mg/dl
    - **0** otherwise
7. **RestingECG**: Resting electrocardiogram results:
    - **Normal**
    - **ST**: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    - **LVH**: showing probable or definite left ventricular hypertrophy by Estes' criteria
8. **MaxHR**: Maximum heart rate achieved, beats per minute.
9. **ExerciseAngina**: Exercise-induced angina, yes or no.
10. **Oldpeak**: Numeric measure of ST depression induced by exercise relative to rest.
11. **ST_Slope**: The slope of the peak exercise ST segment.
    - Up: upsloping
    - Flat
    - Down: downsloping
12. **HeartDisease**: Output class **1**: heart disease, **0**: Normal

## <div style="color:rgb(0, 103, 71);display:fill;border-radius:5px;background-color:whie;letter-spacing:0.1px;overflow:hidden"><p style="padding:10px;color:rgb(0, 103, 71);overflow:hidden;margin:0;font-size:100%; text-align:center"><b id= '002'>Part 2 :</b> Data Prepaeration</p></div>

<h3 style="color:#EE2737FF">Data Loading Hands On</h3>

Data preparation is all about getting raw data ready for analysis and processing. This involves several important steps, like collecting the data, cleaning it to remove errors or inconsistencies, labeling it for better understanding, and organizing it in a way that works well with machine learning (ML) algorithms. Once the data is ready, we explore it and create visualizations to better understand its patterns and insights.


Data wrangling is the process of turning messy, raw data into a clean and organized format that's ready for analysis. This involves two key steps: cleaning the data by removing errors, inconsistencies, or unnecessary information, and structuring it into a clear, consistent format that makes it easier to work with for analysis and modeling.


```python
# Importing the pandas library, which is a powerful tool for data analysis and manipulation in Python.
# Pandas provides structures like DataFrame to handle tabular data (rows and columns) efficiently.
import pandas as pd

# Using pandas to read a CSV (Comma-Separated Values) file into a DataFrame.
# The DataFrame is a 2D structure, like a table in Excel or a SQL database.
# "../assets/heart.csv" is the relative file path to the dataset.
# The resulting DataFrame is stored in a variable named 'df'.
df = pd.read_csv("../assets/heart.csv")
```

- We use **pandas** to easily handle and analyze tabular data.
- **read_csv** loads the data from a file (heart.csv) into a DataFrame so we can work with it like a table.
- The variable **df** stores this data for use in further steps.

```python
# Displays a concise summary of the DataFrame.
# Includes column names, count of non-null values, data types, and memory usage.
# Helpful for understanding data structure and spotting missing or unexpected data types.
df.info()

# Provides descriptive statistics for numerical columns.
# Includes count, mean, standard deviation, minimum, maximum, and percentiles.
# Useful for understanding the distribution and variability of numeric data.
df.describe()

# Provides descriptive statistics for categorical (object) columns.
# Includes count, unique values, top category, and frequency of the top category.
# Useful for understanding the distribution of categorical data.
df.describe(include="object")

# Displays the first 5 rows of the DataFrame (default is 5, but can be changed).
# Useful for a quick preview of the data.
df.head()

# Displays the last 5 rows of the DataFrame (default is 5, but can be changed).
# Useful for verifying the end of the dataset, especially for time-series or ordered data.
df.tail()

# Displays the number of rows and columns in the DataFrame as a tuple (rows, columns).
# Useful for getting the size of the dataset.
df.shape

# Lists all the column names in the DataFrame.
# Helpful for referencing columns and verifying the dataset structure.
df.columns

# Counts the number of unique values for each column in the DataFrame.
# Useful for identifying categorical columns and understanding data diversity.
df.nunique()

# Counts the number of rows with missing values for each column.
# Provides a column-wise summary of missing data in the dataset.
df.isna().sum()

# Checks if any column has at least one missing (NaN) value.
# Useful for quickly identifying if there is missing data in the DataFrame.
df.isna().any()

# Counts the number of columns in the DataFrame that have missing (NaN) values.
# This provides the total count of problematic columns.
df.isna().any().sum()

# Checks for duplicate rows in the DataFrame.
# Returns a Series of boolean values where `True` indicates a duplicate row.
# Helps in identifying and cleaning redundant data.
df.duplicated()

# Removes duplicate rows from the DataFrame.
# Ensures the dataset is free from redundant rows that could skew analysis.
df = df.drop_duplicates()

# Provides a random sample of n rows from the DataFrame (default n is 1).
# Useful for quickly inspecting random entries in the dataset.
df.sample(n=5)

# Displays memory usage of each column in bytes.
# Useful for optimizing memory usage, especially with large datasets.
df.memory_usage()

# Returns the column-wise sum of `True` values for a condition (e.g., missing values).
# Useful for quickly aggregating boolean results across the DataFrame.
df.sum()

# Replaces all missing (NaN) values in the DataFrame with a specified value.
# Useful for handling missing data during cleaning.
df.fillna(value="heart_analysis", inplace=True)

# Drops rows or columns with missing values from the DataFrame.
# Helps remove incomplete data for cleaner analysis.
df.dropna(axis=0, inplace=True)  # axis=0 for rows, axis=1 for columns

```

- These functions collectively help you **explore**, **understand**, and **clean** the dataset.
- They are essential for identifying potential issues like **missing data**, **duplicates**, or **incorrect data types** before starting **analysis** or **modeling**.
- The added functions provide greater flexibility for **data exploration**, **cleaning**, and **preprocessing**.


```python
# Define the target variable
target = 'HeartDisease'

# Separate columns by data types and exclude the target variable
float_columns = [col for col in df.select_dtypes(include='float').columns if col != target]
int_columns = [col for col in df.select_dtypes(include='int').columns if col != target]
categorical_columns = [col for col in df.select_dtypes(include='object').columns if col != target]
        
```

- **select_dtypes**:
    - Used to filter columns based on their data type (float, int, or object for categorical data).

- **columns.tolist()**:
    - Converts the filtered columns into a list for easy manipulation.
    

## <div style="color:rgb(0, 103, 71);display:fill;border-radius:5px;background-color:whie;letter-spacing:0.1px;overflow:hidden"><p style="padding:10px;color:rgb(0, 103, 71);overflow:hidden;margin:0;font-size:100%; text-align:center"><b id= '003'>Part 3 :</b> Data Analysis</p></div>

<h3 style="color:#EE2737FF">Data Analysis Hands On</h3>

**Exploratory Data Analysis (EDA)** is the process of examining and understanding a dataset to uncover patterns, relationships, and insights. It helps identify trends, spot anomalies, and check assumptions using summary statistics and visualizations


```python
# Import Library
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import Plot Config
plt.rcParams['figure.dpi'] = 180
plt.rcParams["figure.figsize"] = (8, 6)
sns.set_theme(rc={
    'axes.facecolor': '#FFFFFF',
    'figure.facecolor': '#FFFFFF',
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'axes.titlesize': '13',
})

# Custom color palette
custom_palette = ['#C6AA97', '#AF4343']  

# Plotting for Float Columns
for col in float_columns:
    ax = sns.boxplot(data=df, x=target, y=col, palette=custom_palette)
    plt.title(f"Boxplot of {col} by {target}")
    plt.xlabel("Heart Disease")
    plt.ylabel(col)
    handles = [
        plt.Line2D([0], [0], color='#C6AA97', lw=4, label='No Heart Disease (0)'),
        plt.Line2D([0], [0], color='#AF4343', lw=4, label='Heart Disease (1)'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

    plt.tight_layout()
    plt.show()

# Plotting for Integer Columns
for col in int_columns:
    ax = sns.barplot(data=df, y=col, x=target, errorbar=None, palette=custom_palette, width=0.6)  
    # Adding hatch pattern, black border, and annotations
    for i, bar in enumerate(ax.patches):
        bar.set_edgecolor('black') 
        bar.set_linewidth(1.5)  
        bar.set_hatch('/' if i % 2 == 0 else 'x')
        # Add annotation
        value = bar.get_height()  
        if value > 0:  
            ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.1f}', 
                    ha='center', va='bottom', fontsize=10)

    plt.title(f"Bar Plot of {col} by {target}")
    plt.xlabel("Heart Disease")
    plt.ylabel(col)
    handles = [
        plt.Line2D([0], [0], color='#C6AA97', lw=4, label='No Heart Disease (0)'),
        plt.Line2D([0], [0], color='#AF4343', lw=4, label='Heart Disease (1)'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.tight_layout() 
    plt.show()

# Plotting for Categorical Columns
for col in categorical_columns:
    ax = sns.countplot(data=df, x=col, hue=target, palette=custom_palette)

    # Adding hatch pattern, black border, and annotations
    for i, bar in enumerate(ax.patches):
        bar.set_edgecolor('black') 
        bar.set_linewidth(1.5) 
        bar.set_hatch('/' if i % 2 == 0 else 'x')  
        # Add annotation
        value = bar.get_height()  
        if value > 0: 
            ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.0f}', 
                    ha='center', va='bottom', fontsize=10)

    plt.title(f"Count Plot of {col} by {target}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(
        title="Heart Disease (0 = No, 1 = Yes)", 
        loc='upper center', bbox_to_anchor=(0.5, -0.2), 
        ncol=2, frameon=False
    )
    plt.xticks()  
    plt.tight_layout() 
    plt.show()


```

    
<h3 style="color:#EE2737FF">Float Columns</h3>

**Boxplots**: Show the distribution of numerical data (e.g., *Age, RestingBP, Cholesterol*) for patients with and without heart disease.

![](../assets/float_box_plot.png)

<h3 style="color:#EE2737FF">Integer Columns</h3>

**Bar Plots**: Compare the proportions of integer-based features (e.g., *FastingBS*) for heart disease and normal cases.

![](../assets/age_bar_plot.png)
![](../assets/resting_bp_box_plot.png)
![](..assets/cholestrol_box_plot.png)
![](..assets/fasting_bs_box_plot.png)
![](..assets/max_hr_box_plot.png)
![](..assets/gender_box_plot.png)

<h3 style="color:#EE2737FF">Categorical Columns</h3>

**Count Plots**: Show how categorical variables (e.g., *ChestPainType, RestingECG, ST_Slope*) are distributed for both classes of heart disease.

![](../assets/chest_pain_type_count_plot.png)
![](../assets/resting_ecg_count_plot.png)
![](../assets/exercise_angina_count_plot.png)
![](../assets/st_slope_count_plot.png)

<h3 style="color:#EE2737FF">Plot Stats</h3>

```python

# Prepare the dataset
plot_df = df.copy()

# Plot 1: Number of Patients
p1 = plot_df.HeartDisease.value_counts()
p1 = p1.rename('count').reset_index().sort_values('count', ascending=False)
x1 = p1['index'].apply(lambda x: 'Heart Disease' if x == 1 else 'No Disease')

# Plot 2: Median Age
p2 = plot_df.groupby('HeartDisease')['Age'].median().round(0).astype(int)
p2 = p2.rename('age').reset_index().sort_values('age', ascending=False)

# Plot 3: Prevalence of Heart Disease by Sex
p3 = plot_df.groupby('Sex')['HeartDisease'].value_counts(normalize=True)
p3 = p3.mul(100).rename('pct').reset_index()
x2 = p3.Sex.apply(lambda x: 'Women' if x == 'F' else 'Men').unique()[::-1]
y0 = p3[p3.HeartDisease == 0]['pct'][::-1]
y1 = p3[p3.HeartDisease == 1]['pct'][::-1]

# Create the subplots layout
fig = plt.figure(figsize=(12, 12))
grid = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1])

# Plot 1: Number of Patients
ax1 = fig.add_subplot(grid[0, 0])
bars1 = ax1.bar(x1, p1['count'], color=['#AF4343', '#C6AA97'], edgecolor='black', linewidth=1.5)
for bar, hatch in zip(bars1, ['x', '/']):  # Add hatches
    bar.set_hatch(hatch)
ax1.set_title("Number of Patients")
ax1.set_ylabel("Count")
ax1.set_xticks(range(len(x1)))
ax1.set_xticklabels(x1, rotation=0)
for i, v in enumerate(p1['count']):
    ax1.text(i, v + 5, f'n = {v}', ha='center')

# Plot 2: Median Age
ax2 = fig.add_subplot(grid[0, 1])
bars2 = ax2.bar(x1, p2['age'], color=['#AF4343', '#C6AA97'], edgecolor='black', linewidth=1.5)
for bar, hatch in zip(bars2, ['x', '/']):  # Add hatches
    bar.set_hatch(hatch)
ax2.set_title("Median Age")
ax2.set_ylabel("Age, in years")
ax2.set_xticks(range(len(x1)))
ax2.set_xticklabels(x1, rotation=0)
for i, v in enumerate(p2['age']):
    ax2.text(i, v + 1, f'{v}', ha='center')

# Plot 3: Prevalence of Heart Disease among Men and Women (full-width plot)
ax3 = fig.add_subplot(grid[1, :])  # Span across both columns
bar_width = 0.35
indices = np.arange(len(x2))
bars3a = ax3.bar(
    indices - bar_width/2, y1, bar_width, 
    label='Heart Disease', color='#AF4343', 
    edgecolor='black', linewidth=1.5
)
bars3b = ax3.bar(
    indices + bar_width/2, y0, bar_width, 
    label='No Disease', color='#C6AA97', 
    edgecolor='black', linewidth=1.5
)
for bar, hatch in zip(bars3a, ['x'] * len(bars3a)):  # Add hatches for Heart Disease
    bar.set_hatch(hatch)
for bar, hatch in zip(bars3b, ['/'] * len(bars3b)):  # Add hatches for No Disease
    bar.set_hatch(hatch)
ax3.set_title("Prevalence of Heart Disease among Men and Women")
ax3.set_ylabel("Proportion (%)")
ax3.set_xticks(indices)
ax3.set_xticklabels(x2, rotation=0)
ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
for i, (v1, v0) in enumerate(zip(y1, y0)):
    ax3.text(i - bar_width/2, v1 + 1, f'{v1:.1f}%', ha='center')
    ax3.text(i + bar_width/2, v0 + 1, f'{v0:.1f}%', ha='center')

# Adjust layout
fig.tight_layout()
plt.subplots_adjust(hspace=0.2)

# Show the plot
plt.show()

```

![](../assets/advanced_analysis_1.png)

In simple terms, the data we’re working with is fairly balanced when it comes to the number of people with and without heart disease. Out of **918** patients, **508** have heart disease, while **410** don’t. The average (median) age of people with **heart disease is 57**, while those without it are slightly younger, with an **average age of 51**. Looking at **gender**, about **63% of men** have heart disease, while for **women**, only about **1** in **4** are diagnosed with it.

<h3 style="color:#EE2737FF">Plot Distributions</h3>


```python
# Define the features to plot and their titles
features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
titles = [
    "Distribution of Age by Heart Disease",
    "Distribution of Systolic Blood Pressure by Heart Disease",
    "Distribution of Cholesterol by Heart Disease",
    "Distribution of Maximum Heart Rate by Heart Disease",
    "Distribution of ST Segment Depression by Heart Disease"
]

# Create a figure with subplots
fig, axes = plt.subplots(len(features), 2, figsize=(16, 25), gridspec_kw={'width_ratios': [2, 1]})
fig.subplots_adjust(hspace=0.4, wspace=0.3)

# Define custom colors
colors = ['#AF4343', '#C6AA97']

# Plot each feature
for i, feature in enumerate(features):
    # Histogram (left column)
    sns.histplot(data=plot_df, x=feature, hue='HeartDisease', stat='density', common_norm=False,
                 palette=colors, edgecolor='black', linewidth=1, ax=axes[i, 0], alpha=0.8)
    axes[i, 0].set_title(titles[i], fontsize=14)
    axes[i, 0].set_xlabel(feature)
    axes[i, 0].set_ylabel("Probability Density")
    axes[i, 0].grid(True, linestyle='--', alpha=0.6)

    # Boxplot (right column)
    sns.boxplot(data=plot_df, y=feature, x='HeartDisease', palette=colors, ax=axes[i, 1])
    axes[i, 1].set_title(titles[i], fontsize=14)
    axes[i, 1].set_xlabel("Heart Disease (0 = No, 1 = Yes)")
    axes[i, 1].set_ylabel(feature)
    axes[i, 1].grid(True, linestyle='--', alpha=0.6)

# Add a shared title for the figure
fig.suptitle("Heart Disease Distributions by Feature", fontsize=18, y=0.91)

# Show the plot
plt.show()

```

![](../assets/advanced_analysis_2.png)

- **Age**: Patients with heart disease tend to fall within a narrower age range, mostly **between 51 and 62 year**s, though a few younger patients stand out as outliers. In contrast, patients without heart disease show a wider range of ages, mostly **between 43 and 57 years**, and no outliers are seen in this group.

- **Systolic Blood Pressure**: Blood pressure levels are similar between the two groups, with most readings falling **between 120 and 145 mmHg**. Both groups have some outliers, but the typical (median) blood pressure is about **130 mmHg for everyone**.

- **Cholesterol**: Cholesterol levels are unevenly spread, especially for patients with heart disease. Many cholesterol values are missing and recorded as 0, which will be fixed during data cleaning. The data for this feature is not well-distributed and skews toward higher values.

- **Heart Rate**: People without heart disease tend to reach higher maximum heart rates compared to those with heart disease. The typical (median) maximum heart rate is **150 beats per minute for those without heart disease**, while it’s **lower at 126 beats per minute for those with heart disease**.

- **ST Segment Depression (OldPeak)**: There’s a noticeable difference in this measurement between the groups. For heart disease patients, the range of values is wider, mostly between 0 and 2 mm, with some large outliers and a typical value (median) of **1.2 mm**. For patients without heart disease, the range is narrower, mostly between 0 and **0.6 mm**, with a median of **0 mm**, but the data for this group is unevenly spread.


## <div style="color:rgb(0, 103, 71);display:fill;border-radius:5px;background-color:whie;letter-spacing:0.1px;overflow:hidden"><p style="padding:10px;color:rgb(0, 103, 71);overflow:hidden;margin:0;font-size:100%; text-align:center"><b id= '004'>Part 4 :</b> Machine Learning Modeling</p></div>

<h3 style="color:#EE2737FF">Spliting Data into Train, Validation, & Test</h3>

- **Library Imports**:
    - **Pipeline**: A way to chain multiple processing steps together.
    - **ColumnTransformer**: A tool to handle different types of columns (e.g., categorical, numerical) with different transformations.
    - **Train-Test Split**: To divide data into training, validation, and testing sets.
    - **Encoders and Scalers**: For preprocessing data.

- **Separate Predictors** (*X*) and Target (*y*):
    - **X** contains **all** the columns except the **target (HeartDisease)**, and **y** contains the **target** values (HeartDisease).

- **Divide Data**:
    - **Training Data** (**60% of the total**): Used to train the model.
    - **Validation** and **Testing** Data (**20% each**):
        - Validation data helps tune the model during training.
        - Testing data evaluates the model after training.
        
- **Select Column Types**:
    - **Categorical Columns**: Columns with string/text data (*e.g., "Male", "Yes"*).
    - **Numerical Columns**: Columns with numeric data (*e.g., age, blood pressure*).

- **Preprocessing**:
    - For **Categorical Data**:
        - Used **OneHotEncoder** to convert text labels into numeric values.
        - Example: "Male" → [1, 0], "Female" → [0, 1].
    - For **Numerical** Data:
        - Used StandardScaler to normalize the data.
        - Ensures all numbers are scaled consistently (*e.g., mean = 0, standard deviation = 1*).
        
- **Combine Preprocessing**:
    - **ColumnTransformer**: Applied different preprocessing steps to numerical and categorical columns.
    - Makes sure each column is **prepared** properly before feeding the data to a machine learning model.

```python
# Import Library
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Separate target from predictors
X = df.drop(['HeartDisease'], axis=1)
y = df['HeartDisease']


# Divide data into training, validation, and Testing subsets
# In the first step we will split the data in training and remaining dataset
X_train_full, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.6)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
X_valid_full, X_test_full, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)


# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [
    cname for cname in X_train_full.columns
    if X_train_full[cname].dtype == "object"
]


# Select numerical columns
numerical_cols = [
    cname for cname in X_train_full.columns 
    if X_train_full[cname].dtype in ['int64', 'float64']
]


# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Preprocessing for numerical data
numerical_transformer = Pipeline([
    ('std_scaler', StandardScaler())
])


# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

```

Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier. For each model, we follow a structured process using pipelines to streamline the workflow by combining data preprocessing and model fitting steps.

<h3 style="color:#EE2737FF">Logistic Regression</h3>

- A simple linear model that predicts probabilities for the target variable (HeartDisease) based on the relationship between input features. Logistic regression is commonly used for binary classification problems.
- We first define the model (logistic_regression) and integrate it with the preprocessing pipeline (preprocessor), ensuring the categorical and numerical data are appropriately transformed before being passed to the model.
- The pipeline is then fitted on the training data (X_train and y_train), allowing the model to learn patterns in the data.
- Using the trained pipeline, predictions (train_pred_logistic_regression) and prediction probabilities (train_proba_logistic_regression) are generated for the training, validation, and test datasets, ensuring consistency across all data splits.

```python
# Simple Model Creation
logistic_regression = LogisticRegression()


# Create Pipeline
logistic_regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', logistic_regression)
])

# Preprocessing of training data, fit model 
logistic_regression_pipeline.fit(X_train, y_train)

# Preprocessing of training data, get predictions
train_pred_logistic_regression= logistic_regression_pipeline.predict(X_train)
train_proba_logistic_regression = logistic_regression_pipeline.predict_proba(X_train)[:, 1]

# Preprocessing of validation data, get predictions
valid_pred_logistic_regression = logistic_regression_pipeline.predict(X_valid)
valid_proba_logistic_regression = logistic_regression_pipeline.predict_proba(X_valid)[:, 1]

# Preprocessing of validation data, get predictions
test_pred_logistic_regression = logistic_regression_pipeline.predict(X_test)
test_proba_logistic_regression = logistic_regression_pipeline.predict_proba(X_test)[:, 1]
```

<h3 style="color:#EE2737FF">Random Forest Classifier</h3>

- A more complex model based on decision trees, where multiple trees are trained on different data subsets and combined to improve accuracy and reduce overfitting.
- Like logistic regression, we define the model (random_forest) and embed it in a pipeline with the preprocessor.
- The model is trained on the training data, and predictions and probabilities are obtained for the training, validation, and test datasets. Random forests handle non-linear relationships and interactions well, making them robust for various data distributions.

```python
# Simple Model Creation
random_forest = RandomForestClassifier()


# Create Pipeline
random_forest_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', random_forest)
])

# Preprocessing of training data, fit model 
random_forest_pipeline.fit(X_train, y_train)

# Preprocessing of training data, get predictions
train_pred_random_forest = random_forest_pipeline.predict(X_train)
train_proba_random_forest = random_forest_pipeline.predict_proba(X_train)[:, 1]

# Preprocessing of validation data, get predictions
valid_pred_random_forest = random_forest_pipeline.predict(X_valid)
valid_proba_random_forest = random_forest_pipeline.predict_proba(X_valid)[:, 1]

# Preprocessing of validation data, get predictions
test_pred_random_forest = random_forest_pipeline.predict(X_test)
test_proba_random_forest = random_forest_pipeline.predict_proba(X_test)[:, 1]
```

<h3 style="color:#EE2737FF">Gradient Boosting Classifier</h3>

- An advanced model that builds trees sequentially, each focusing on correcting the mistakes of the previous one. This method often yields better performance, especially on more complex datasets.
- Similar to the previous models, we define the gradient boosting classifier (gradient_boosting), include it in a pipeline with the preprocessor, and fit it on the training data.
- Predictions and probabilities are generated for all datasets, just as with the other models.


```python
# Simple Model Creation
gradient_boosting = GradientBoostingClassifier()


# Create Pipeline
gradient_boosting_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', gradient_boosting)
])

# Preprocessing of training data, fit model 
gradient_boosting_pipeline.fit(X_train, y_train)

# Preprocessing of training data, get predictions
train_pred_gradient_boosting = gradient_boosting_pipeline.predict(X_train)
train_proba_gradient_boosting = gradient_boosting_pipeline.predict_proba(X_train)[:, 1]

# Preprocessing of validation data, get predictions
valid_pred_gradient_boosting = gradient_boosting_pipeline.predict(X_valid)
valid_proba_gradient_boosting = gradient_boosting_pipeline.predict_proba(X_valid)[:, 1]

# Preprocessing of validation data, get predictions
test_pred_gradient_boosting = gradient_boosting_pipeline.predict(X_test)
test_proba_gradient_boosting = gradient_boosting_pipeline.predict_proba(X_test)[:, 1]
```


The preprocessing step ensures all datasets (training, validation, and test) are consistently transformed using the same pipeline, preventing data leakage and maintaining model robustness. By structuring the workflow with pipelines, we eliminate repetitive code and guarantee seamless integration of preprocessing and modeling. Each model provides predictions and probabilities for further evaluation, which will help us compare their performance and determine the best-suited model for the dataset. Next, we will evaluate these predictions using metrics like accuracy, ROC curves, and confusion matrices to identify the most effective model.


## <div style="color:rgb(0, 103, 71);display:fill;border-radius:5px;background-color:whie;letter-spacing:0.1px;overflow:hidden"><p style="padding:10px;color:rgb(0, 103, 71);overflow:hidden;margin:0;font-size:100%; text-align:center"><b id= '004'>Part 5 :</b> Model Evaluation & Interpretation</p></div>


The function `plot_evaluation` evaluates a machine learning model by calculating key metrics like accuracy, a confusion matrix, a classification report, and the AUC score. It displays these metrics both as printed text and visual plots. The confusion matrix shows where the model gets predictions right or wrong, while the ROC curve illustrates the balance between true positives and false positives, with the AUC score summarizing the model's overall performance. This provides a clear, side-by-side view of the model's effectiveness for easy understanding.


```python
# Import Library
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report

def plot_evaluation(true, predicted, probabilities, dataset_name):
    """
    Plots the confusion matrix and ROC curve side by side and prints evaluation metrics.
    """
    # Compute metrics
    acc = accuracy_score(true, predicted)
    cm = confusion_matrix(true, predicted)
    report = classification_report(true, predicted, output_dict=False)
    roc_auc = roc_auc_score(true, probabilities)
    fpr, tpr, _ = roc_curve(true, probabilities)

    # Print Evaluation Metrics
    print(
        colored("==>", attrs=["bold"]),
        colored(f" {dataset_name} Dataset Evaluation ", "cyan", attrs=["bold", "reverse", "blink"]),
    )
    print(
        colored("==>", attrs=["bold"]),
        colored(" Model Accuracy: ", "green", attrs=["bold"]),
        colored(f"{acc * 100:.2f}%", attrs=["bold"]),
    )
    print(colored("\n\n Confusion Matrix: ", "blue", attrs=["bold", 'reverse', 'blink']))
    print(colored("\n Classification Report: ", "red", attrs=["bold", 'reverse', 'blink']), '\n')
    print(colored(report, 'yellow', attrs=['bold']))
    print(colored(38 * '*=', attrs=['bold']))
    print('\n')

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Confusion Matrix
    sns.heatmap(cm, annot=True, fmt=".0f", cmap='Reds', ax=ax[0], cbar=False)
    ax[0].set_title(f"{dataset_name} Confusion Matrix")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")

    # Plot ROC Curve
    ax[1].plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax[1].set_title(f"{dataset_name} ROC Curve")
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].legend(loc="lower right")

    # Adjust layout
    plt.tight_layout()
    plt.show()
  
```

<h3 style="color:#EE2737FF">Logistic Regression Evaluation</h3>

```python
# Evalaution on Training Dataset
plot_evaluation_with_roc(
    y_train, train_pred_logistic_regression, 
    train_proba_logistic_regression, "Training"
)
```

![](../assets/lr_train_acc.png)
![](../assets/lr_train.png)

```python
# Evalaution on Validation Dataset
plot_evaluation_with_roc(
    y_valid, valid_pred_logistic_regression, 
    valid_proba_logistic_regression, "Validation"
)
```

![](../assets/lr_valid_acc.png)
![](../assets/lr_valid.png)


```python
# Evalaution on Test Dataset
plot_evaluation_with_roc(
    y_test, test_pred_logistic_regression, 
    test_proba_logistic_regression, "Test"
)

```

![](../assets/lr_test_acc.png)
![](../assets/lr_test.png)


<h3 style="color:#EE2737FF">RandomForest Classifier</h3>

```python
# Evalaution on Training Dataset
plot_evaluation_with_roc(
    y_train, train_pred_random_forest, 
    train_proba_random_forest, "Training"
)
```

![](../assets/rf_train_acc.png)
![](../assets/rf_train.png)

```python
# Evalaution on Validation Dataset
plot_evaluation_with_roc(
    y_valid, valid_pred_random_forest, 
    valid_proba_random_forest, "Validation"
)
```

![](../assets/rf_valid_acc.png)
![](../assets/rf_valid.png)


```python
# Evalaution on Test Dataset
plot_evaluation_with_roc(
    y_test, test_pred_random_forest, 
    valid_proba_random_forest, "Test"
)

```

![](../assets/rf_test_acc.png)
![](../assets/rf_test.png)


<h3 style="color:#EE2737FF">GradientBoosting Classifier</h3>

```python
# Evalaution on Training Dataset
plot_evaluation_with_roc(
    y_train, train_pred_gradient_boosting, 
    train_proba_gradient_boosting, "Training"
)
```

![](../assets/gb_train_acc.png)
![](../assets/gb_train.png)

```python
# Evalaution on Validation Dataset
plot_evaluation_with_roc(
    y_valid, valid_pred_gradient_boosting, 
    valid_proba_gradient_boosting, "Validation"
)
```

![](../assets/gb_valid_acc.png)
![](../assets/gb_valid.png)


```python
# Evalaution on Test Dataset
plot_evaluation_with_roc(
    y_test, test_pred_gradient_boosting, 
    test_proba_gradient_boosting, "Test"
)

```

![](../assets/gb_test_acc.png)
![](../assets/gb_test.png)


## <div style="color:rgb(0, 103, 71);display:fill;border-radius:5px;background-color:whie;letter-spacing:0.1px;overflow:hidden"><p style="padding:10px;color:rgb(0, 103, 71);overflow:hidden;margin:0;font-size:100%; text-align:center"><b id= '006'>Part 6 :</b> References</p></div>


<h3 style="color:#EE2737FF">References</h3>

- [Sk-Learn](https://scikit-learn.org/dev/index.html)
- Lundberg, Scott M., and Lee, Su-In. "A unified approach to interpreting model predictions." Advances in Neural Information Processing Systems. (2017).
- Virani, S. S., Alonso, A., Aparicio, H. J., Benjamin, E. J., Bittencourt, M. S., Callaway, C. W., Carson, A. P., Chamberlain, A. M., Cheng, S., Delling, F. N., Elkind, M. S. V., Evenson, K. R., Ferguson, J. F., Gupta, D. K., Khan, S. S., Kissela, B. M., Knutson, K. L., Lee, C. D., Lewis, T. T., … Tsao, C. W. (2021). Heart disease and stroke statistics—2021 update. Circulation, 143(8).
- World Health Organization. Cardiovascular diseases (CVDs). World Health Organization. (2021, June 11).

