# 🧠 Customer Segmentation using PySpark -Week05

This project demonstrates how to perform **multi-class classification** on a customer dataset using **Apache Spark (PySpark)**. The goal is to classify customers into different segments based on their behavior and demographics using machine learning models like **Decision Tree** and **Logistic Regression (One-vs-Rest)**.

---

## 📁 Dataset

The dataset is located at: drive/My Drive/Colab Notebooks/customer_segmentation_dataset.csv

### Features:
- `age`: Age of the customer
- `annual_income`: Annual income in USD
- `spending_score`: Score assigned based on spending behavior
- `years_as_customer`: Number of years with the company
- `number_of_purchases`: Total purchases made
- `customer_segment`: Target label (Segment category)

---

## 🔧 Technologies Used

- Python
- Google Colab
- PySpark (Spark MLlib)
- Pandas, Matplotlib, Seaborn

---

## 📊 Workflow

### 1. Environment Setup
- PySpark installation
- Mount Google Drive to access the dataset

### 2. Data Loading & Inspection
- Load CSV into a PySpark DataFrame
- Infer schema & show summary of data
- Display distinct `customer_segment` classes

### 3. Data Preparation
- Assemble features into a single vector using `VectorAssembler`
- Drop unnecessary columns after transformation
- Encode `customer_segment` using `StringIndexer`

### 4. Decision Tree Classification
- Split data into training (70%) and test (30%)
- Train `DecisionTreeClassifier` with `entropy` impurity and `maxDepth=4`
- Evaluate model accuracy using `MulticlassClassificationEvaluator`
- Display model structure using `.toDebugString`

### 5. Logistic Regression (One-vs-Rest)
- Apply One-vs-Rest strategy using `LogisticRegression`
- Fit model and predict on test set
- Evaluate performance
- Generate and visualize **Confusion Matrix** using Seaborn

---



## 📈 Confusion Matrix

Visual representation of the classification results using Logistic Regression (OvR):

![Confusion Matrix](#)

---

## 📦 How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/customer-segmentation-pyspark.git
    ```

2. Open `customer_segmentation.ipynb` in **Google Colab**

3. Mount Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

4. Install dependencies (only needed in Colab):
    ```bash
    !pip install pyspark
    ```

5. Run all cells in sequence

---


## 🙋‍♂️ Author

**M. Abinath**  
MSc Artificial Intelligence  
University of East London

---



