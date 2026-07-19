# Iris Flower Classification with Machine Learning

This project implements a data science and machine learning workflow to classify different species of Iris flowers (Setosa, Versicolor, and Virginica) based on their structural physical dimensions. The data analysis pipeline and classification models are built and optimized inside a Google Colab notebook environment using a localized `Iris.csv` file.

## 🚀 Project Overview
The Iris dataset is a classic algorithmic benchmarking tool where physical properties—specifically sepal length, sepal width, petal length, and petal width—are analyzed to cluster morphological targets. This project extracts, cleanses, visualizes, and scales these geometric values before feeding them into a optimized K-Nearest Neighbors (KNN) model structure to output automated taxonomy predictions.

## 🛠️ Tech Stack & Libraries
* **Environment:** Google Colab / Jupyter Notebook
* **Languages:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn

## 📂 Project Structure & Workflow

### Step 1: Workspace Initialization
Set up the localized project environment fields before introducing structural code logic blocks.

### Step 2: Library Setup
Import open-source structural packages (`pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`) required to process multidimensional data matrices.

### Step 3: Data Cleansing & Extraction
The dataset file (`Iris.csv`) is read into a structured DataFrame. Unnecessary structural identifier columns (such as index `Id` columns) are dropped to streamline pattern training, and rows are cleaned using null drops.

### Step 4: Cluster Exploratory Visualization
Generates an advanced categorical scatter plot matrix (Pairplot). This visualization charts feature pairings simultaneously, highlighting how petal and sepal sizes form clear boundaries between individual plant types.

### Step 5: Feature Engineering & Partitioning
* **Target Isolation:** Independent measurement attributes are decoupled from target output strings (`Species`).
* **Train/Test Splitting:** The data is systematically separated using an **80% Training / 20% Evaluation split**, using stratification to guarantee uniform species balances across partitions.
* **Standard Scaling:** Column arrays are normalized using standard deviations to ensure all measurements sit on equal mathematical weights.

### Step 6: Predictive Model Construction
A K-Nearest Neighbors (KNN) distance-metric algorithm is built and trained on the scaled training datasets.

### Step 7: Granular Metric Review
The hidden evaluation partitions are evaluated to generate accuracy performance scores alongside precision, recall, and F1-score tracking.

### Step 8: Error Analysis Matrix
Generates a second visual chart using a confusion matrix heatmap. This visual highlights true positive classifications versus any edge-case misclassifications.

---

## 📈 Key Visual Outputs
The workflow generates two critical visualization charts within your Google Colab sheet:
1. **Feature Pairplot Matrix:** Identifies specific feature combinations (such as Petal Length vs. Petal Width) that exhibit strong clustering separation.
2. **Confusion Matrix Heatmap:** Displays model performance metrics visually across all three structural species targets.

## 💻 How to Run This Project
1. Open a new workspace in [Google Colab](https://google.com).
2. Open the **Folder icon** 📁 on the left menu pane and upload your dataset file (`Iris.csv`).
3. Run through each individual code block cell sequentially from Step 2 to Step 8.
