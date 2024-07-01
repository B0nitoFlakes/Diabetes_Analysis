import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import warnings

# Feature Selections
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2

# Train test splits
from sklearn.model_selection import train_test_split

# Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

# AUC accuracy score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Colormap module 
import matplotlib.cm as cm  

st.set_option('deprecation.showPyplotGlobalUse', False)

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_csv("/Users/HP/Documents/UOW/JAN SEM 2024/Data Visualization/ASSIGNMENT2/Dataset/diabetes.csv")


def main():
    page = st.sidebar.selectbox("Go to", ["Home", "Dataset", "Data Distribution", "Data Correlation", "Feature Selection", "Predictions", "Conclusion"])

    if page == "Home":
        show_homepage()
    elif page == "Dataset":
        show_datasetpage()
    elif page == "Data Distribution":
        show_data_distributionpage()
    elif page == "Data Correlation":
        show_datacorrelationpage()
    elif page == "Feature Selection":
        show_featureselectionpage()
    elif page == "Predictions":
        show_predictionpage()
    elif page == "Conclusion":
        show_conclusionpage()

def show_homepage():
    st.title("Diabetes Outcome Analysis")

    st.markdown("### By: Marco Setiawan")
    st.markdown("### Student ID: 0134172")
    
    st.write("""
    ## Objective:
    This Streamlit app aims to provide an interactive analysis of diabetes outcomes using various visualization techniques and machine learning models. 
    """)

    st.write("""
    ## List of Questions:
    These are the questions that we want to tackle through our data visualizations later on

    * Do pregnancies increase the posibility of someone getting diabetes?
    * Does glucose level affects the chances of someone getting diabetes?
    * Does higher blood preasure means that they can get diabetes or vice versa?
    * Does skin thickness affects the chances of someone getting diabetes?
    * Does Insulin levels affects the chances of someone getting diabetes?
    * Does higher BMI caused resulting in higher chances of developing diabetes?
    * Does Age matters in terms of the chances of developing diabetes?
    * Does Diabetes Pedigree Functions affects the chances of developing diabetes?
    """)

def show_datasetpage():
    global data
    st.title("Dataset Page")

    st.write("""
    ## Introduction into Diabetes:
    Diabetes is a long-term (chronic) illness that affects how your body uses food as fuel.

    The majority of the food you eat is converted by your body into sugar, or glucose, which is then released into your bloodstream. Your pancreas releases insulin in response to an increase in blood sugar. Insulin functions as a key to allow blood sugar to enter your body's cells and be used as an energy source.

    When you have diabetes, your body is either unable to produce enough insulin or is unable to use it effectively. Too much blood sugar remains in your circulation when there is insufficient insulin or when cells cease reacting to insulin. That can eventually lead to major health issues like renal disease, heart disease, and eyesight loss.
    """)
    # You can add code here to display your dataset or any other content for the dataset page
    
    st.markdown("### Dataset")
    # Display the data
    st.write(data)

    st.markdown("### Dataset Description")
    st.write(data.describe())

    st.markdown("### Checking zero values in the Dataset")
    # Check if there is any 0 values in the dataset
    for column in data.columns:
        # Check if the column contains the value '0' and is not null
        zero_not_null = data[(data[column] == 0)]
        if not zero_not_null.empty:
            st.write(f"{column}: {zero_not_null.shape[0]}")

    st.write("So, what we have discovered in the cell above tells us that there are alot of 0 values in features that are not supposed to get 0 such as Blood Pressure, Skin Thickness, Insulin, BMI, and Glucose. Therefore, we decided to change the 0 values into their respective mean values for more accurate prediction later on")
    
    # List of columns where you want to replace zero values with means
    columns_to_process = ['Glucose', 'BMI', 'Insulin','SkinThickness','BloodPressure',]  # Add more columns as needed

    # Replace zero values with mean values for specified columns
    for column in columns_to_process:
        # Calculate the mean excluding zeros
        mean_without_zeros = data[data[column] != 0][column].mean()
        
        # Replace zeros with the calculated mean
        data.loc[data[column] == 0, column] = mean_without_zeros

    st.markdown("### Data description after changing zero to mean")
    st.write(data.describe())

    st.markdown("### Outlier Detection")
    # Function to calculate Z-scores for a dataset

    def calculate_z_scores(column):
        mean = np.mean(column)
        std_dev = np.std(column)
        z_scores = (column - mean) / std_dev
        return z_scores

    # Calculate Z-scores for each class (assuming each class is a column in your DataFrame)
    z_scores = data.apply(calculate_z_scores)

    # Define marker for outliers
    outlier_marker = 'x'

    # Set up the layout for subplots
    num_rows = 3
    num_cols = 3
    num_plots = num_rows * num_cols

    # Define colors for each class
    colors = ['#1f77b4', '#4b89dc', 'lightblue',   # Shades of Blue
            '#d62728', 'brown', 'orange',   # Shades of Red
            '#234F1E', '#29AB87', '#aed581']   # Shades of Green


    # Iterate through each column (class) in the DataFrame and create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Iterate through each column (class) in the DataFrame
    for i, column in enumerate(z_scores.columns):
        # Scatter plot for the current class
        axes[i].scatter(data.index, z_scores[column], label=column, color=colors[i % len(colors)])

        # Identify outliers (using a simple threshold for demonstration)
        outliers = z_scores[z_scores[column].abs() > 5]  # Using 5 standard deviations as threshold for outliers
        axes[i].scatter(outliers.index, outliers[column], marker=outlier_marker, color='black', label='Outliers')

        axes[i].set_title(f'Z-score Distribution for {column}')
        axes[i].set_ylabel('Z-score')
        axes[i].set_xlabel('Data Point Index')
        axes[i].legend()
        axes[i].grid(True)

    # Remove any unused subplots
    for j in range(i + 1, num_plots):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot()

    st.write("z-score is utilized in here to display the outliers among the features above, and with a threshold of 5, all the data that go over 5 will be identified as outliers. Therefore, the x in each of the graphs provide a clear vision on which data points in the scatter plot are the outliers.")

    # Set the threshold for identifying outliers
    z_score_threshold = 5  # Using 5 standard deviations as threshold for outliers

    # Filter out outliers based on the threshold
    data = data[(z_scores.abs() <= z_score_threshold).all(axis=1)]

    st.write("## Data description after removing outliers")
    st.write(data.describe())

def show_data_distributionpage():
    st.title("Data Distribution Page")

    # Plot histograms for each column in a 3 by 3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Flatten the axes array
    axes = axes.flatten()

    # Iterate over each column and plot its histogram
    for i, column in enumerate(data.columns):
        ax = axes[i]  # Get the current axis
        ax.hist(data[column], bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f'Histogram of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.grid(True)

    # Remove empty subplots
    for i in range(len(data.columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot()

    st.write("Based on the graphs plotted above, we can discover data spread of each features. Additionally, we can uncover unique insight such as the histogram of insulin that has a sudden spike around the 140 to 160 range which has the highest frequency of more than 400. The same goes to Skin Thickness that has the frequency of more than 250 for the thickness around 28 to 32. On the contrary there are also some that are normally distributed such as Glucose, Blood Pressure, and BMI which means that the mean, median, and mode of the distribution are all equal, and they are located at the center of the curve. Age, Pregnancies, and Diabetes Pedigree Function has a positive skewed histogram, meaning that the data is more heavily weighted towards the lower end of the range, and has a mean that is higher than the median.")

def show_datacorrelationpage():
    global data
    st.title("Data Correlation")

    st.write("## Data Correlation via Heatmap")
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='Reds', fmt=".2f")
    plt.title('Correlation Heatmap')
    st.pyplot()

    st.markdown("## The correlation of Pregnancies with Outcome")
    # create a 2 seperate dataset for outcome 0 and 1
    data_outcome_0 = data[data["Outcome"] == 0]
    data_outcome_1 = data[data["Outcome"] == 1]

    # Create subplots for Outcome 0 and Outcome 1
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Density plot for Outcome 0
    sns.kdeplot(data=data_outcome_0, x='Pregnancies', fill=True, color='red', ax=axs[0])
    axs[0].set_title('Density Plot of Pregnancies for Outcome 0', fontsize=16)
    axs[0].set_xlabel('Pregnancies', fontsize=14)
    axs[0].set_ylabel('Density', fontsize=14)
    axs[0].tick_params(axis='x', labelsize=12)
    axs[0].tick_params(axis='y', labelsize=12)

    # Density plot for Outcome 1
    sns.kdeplot(data=data_outcome_1, x='Pregnancies', fill=True, color='orange', ax=axs[1])
    axs[1].set_title('Density Plot of Pregnancies for Outcome 1', fontsize=16)
    axs[1].set_xlabel('Pregnancies', fontsize=14)
    axs[1].set_ylabel('Density', fontsize=14)
    axs[1].tick_params(axis='x', labelsize=12)
    axs[1].tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    st.pyplot()

    st.markdown("## The correlation of Glucose with Outcome")
    # Set style
    sns.set_style("whitegrid")

    # Create a violin plot with inner box plot
    sns.violinplot(data=data, x='Outcome', y='Glucose', inner='box', linewidth=0, palette="tab10")  # Changing the palette to "tab10"
    plt.title('Violin Plot of Glucose by Outcome', fontsize=16)
    plt.xlabel('Outcome', fontsize=14)
    plt.ylabel('Glucose', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    st.pyplot()

    st.markdown("## The correlation of BMI with Outcome")
    # Set style
    sns.set_style("whitegrid")

    # Create a box plot
    sns.boxplot(data=data, x='Outcome', y='BMI', linewidth=2, palette="Set2")
    plt.title('Box Plot of BMI by Outcome', fontsize=16)
    plt.xlabel('Outcome', fontsize=14)
    plt.ylabel('BMI', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    st.pyplot()

    st.markdown("## The correlation of Age with Outcome")
    # Set style
    sns.set_style("whitegrid")

    # Create separate histograms for Age by Outcome
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    sns.histplot(data=data[data['Outcome'] == 0], x='Age', color='blue', kde=True, ax=axes[0])
    axes[0].set_title('Outcome 0', fontsize=16)
    axes[0].set_xlabel('Age', fontsize=14)
    axes[0].set_ylabel('Frequency', fontsize=14)
    axes[0].tick_params(labelsize=12)

    sns.histplot(data=data[data['Outcome'] == 1], x='Age', color='orange', kde=True, ax=axes[1])
    axes[1].set_title('Outcome 1', fontsize=16)
    axes[1].set_xlabel('Age', fontsize=14)
    axes[1].tick_params(labelsize=12)

    plt.tight_layout()
    st.pyplot()

    st.markdown("## The correlation of Blood Pressure with Outcome")
    # Set style
    sns.set_style("whitegrid")

    # Define custom colors for the box plot
    colors = ['#FF6347', '#87CEEB']  # Coral for Outcome 0, Sky Blue for Outcome 1

    # Create a box plot with custom colors
    sns.boxplot(data=data, x='Outcome', y='BloodPressure', linewidth=2, palette=colors)
    plt.title('Box Plot of Blood Pressure by Outcome', fontsize=16)
    plt.xlabel('Outcome', fontsize=14)
    plt.ylabel('Blood Pressure', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    st.pyplot()

    st.markdown("## The correlation of Skin Thickness with Outcome")
    # Set style
    sns.set_style("whitegrid")

    # Create a violin plot with inner box plot and customized colors
    sns.violinplot(data=data, x='Outcome', y='SkinThickness', inner='box', linewidth=1.5, palette="husl")  
    # Adjust the linewidth and change the palette to "husl" for more colorful plots

    plt.title('Violin Plot with Box Plot of SkinThickness by Outcome', fontsize=16)
    plt.xlabel('Outcome', fontsize=14)
    plt.ylabel('SkinThickness', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    st.pyplot()

    st.markdown("## The correlation of Insulin with Outcome")
    # Create subplots for Outcome 0 and Outcome 1
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Density plot for Outcome 0
    sns.kdeplot(data=data_outcome_0, x='Insulin', fill=True, color='#1f77b4', ax=axs[0])
    axs[0].set_title('Density Plot of Insulin for Outcome 0', fontsize=16)
    axs[0].set_xlabel('Insulin', fontsize=14)
    axs[0].set_ylabel('Density', fontsize=14)
    axs[0].tick_params(axis='x', labelsize=12)
    axs[0].tick_params(axis='y', labelsize=12)

    # Density plot for Outcome 1
    sns.kdeplot(data=data_outcome_1, x='Insulin', fill=True, color='#2ca02c', ax=axs[1])
    axs[1].set_title('Density Plot of Insulin for Outcome 1', fontsize=16)
    axs[1].set_xlabel('Insulin', fontsize=14)
    axs[1].set_ylabel('Density', fontsize=14)
    axs[1].tick_params(axis='x', labelsize=12)
    axs[1].tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    st.pyplot()

def show_featureselectionpage():
    global data
    st.title("Feature Selection")

    st.markdown("## Feature Selection with RandomForestClassification")
    # Dependent variable
    y = data['Outcome']

    # Extract all other columns except 'Outcome' into X
    X = data.drop('Outcome', axis=1)  # Drop the 'Outcome' column from X

    # Instantiate the RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the RandomForestClassifier to your data
    rf.fit(X, y)

    # Get feature importances
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)

    # Sort feature importances in descending order
    sorted_feature_importances = feature_importances.sort_values(ascending=False)

    # Plot the feature importances as a pie chart with colormap
    plt.figure(figsize=(7, 7))
    colors = cm.tab20c(range(len(sorted_feature_importances)))  # Using tab20c colormap for colors
    plt.pie(sorted_feature_importances, labels=sorted_feature_importances.index, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Feature Importances from Random Forest', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()
    st.pyplot()

    st.markdown("## Feature Selection with SelectKBest")
    k = 5  # Number of top features to select
    selector = SelectKBest(score_func=chi2, k=k)
    X_new = selector.fit_transform(X, y)

    # Get the selected feature scores
    feature_scores = selector.scores_

    # Sort feature scores in descending order and get corresponding indices
    sorted_indices = feature_scores.argsort()[::-1][:k]

    # Get the names of the selected features based on sorted indices
    selected_features = X.columns[sorted_indices]

    # Create a DataFrame with the selected features and their scores
    selected_features_df = pd.DataFrame({'Feature': selected_features, 'Score': feature_scores[sorted_indices]})

    # Sort the DataFrame by scores in descending order
    sorted_features_df = selected_features_df.sort_values(by='Score', ascending=False)

    # Define colors for each bar
    num_features = len(sorted_features_df)
    colors = plt.cm.viridis(np.linspace(0, 1, num_features))  # Viridis colormap for a range of colors

    # Create horizontal bar chart with different colors for each bar
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_features_df['Feature'], sorted_features_df['Score'], color=colors)

    # Add labels and title
    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title('Feature Importance from SelectKBest', fontsize=16)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add grid lines for better readability

    # Add legend for the colors
    plt.legend(bars, sorted_features_df['Feature'], bbox_to_anchor=(1.05, 1), loc='upper left')

    st.pyplot()

def show_predictionpage():
    # This function are mostly get from saved images and csv files due to the model accuracy changed when run it in streamlit
    # To prevent the data from jupyter notebook change when run in streamlit, i saved the data from jupyter then upload it here
    st.title("Predictions")

    st.markdown("## AUC accuracy comparison among various classification models")
    st.write("We compared multiple models such as LogisticRegresstion, RandomForestClassification, NaiveBayes, SupportVectorMachine, K-Nearest Neighbour, and also DecisionTreeClassifier")
    data_compare = pd.read_csv("/Users/HP/Documents/UOW/JAN SEM 2024/Data Visualization/ASSIGNMENT2/auc_scores.csv")
    st.write(data_compare)

    st.write("As what we can conclude from this table, logistic regression has the highest AUC score with 0.87, follow with Naive Bayes with 0.86, Random Forest and Support Vector Machine with 0.85, K-Nearest neighbour with 0.79, and Decision Tree with 0.68. Therefore we will use Logistic Regression as our main prediction model")


    st.markdown("## Fine Tune Logistic Regression Model")
    st.write("Accuracy Results:")

    results_df = pd.read_csv('/Users/HP/Documents/UOW/JAN SEM 2024/Data Visualization/ASSIGNMENT2/accuracy_results.csv')
    st.write(results_df)
    st.write("confusion matrix:")
    plot_file_path = "/Users/HP/Documents/UOW/JAN SEM 2024/Data Visualization/ASSIGNMENT2/confusion_matrix.jpeg"
    st.image(plot_file_path)

def show_conclusionpage():
    st.title("Conclusion")
    st.write("We have discovered that there are various feature that can affect the diabetes outcomes, which are BMI, Glucose, Insulin, Pregnancies, and also Age. Although the correlations visualization suggest otherwise, the feature selections were highly recommended these features as the independent variables. Additionally, Skin Thickness, Blood Pressure, and Diabetes Pedigree Function are not the determining factors which are also shown in the correlation visualizations above. Moreover, among all the classification algorithms, logistic regression has the highest accuracy at 0.87%. Thus, this model is chosen to be the model for prediction.")

if __name__ == "__main__":
    main()
