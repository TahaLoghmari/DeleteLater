### 1. Setup & Imports (Cell 1)

- **What it does:** Adds the project's source folder to the system path so we can import custom helper functions from pyspark_utils.py. It then initializes a Spark session.
- **Expected Output:** A SparkSession object summary (version, app name, master URL).

### 2. Title & Description (Cell 2)

- **What it does:** Markdown text describing the project's goal: predicting Alzheimer's disease using Spark MLlib.
- **Expected Output:** Formatted text.

### 3. Library Imports (Cell 3)

- **What it does:** Imports all necessary Python libraries (Pandas, NumPy, Matplotlib, Seaborn) and Spark MLlib modules (Classification, Evaluation, Tuning). It also re-initializes the Spark session to ensure it's ready.
- **Expected Output:** A SparkSession object summary.

### 4. Load Dataset Header (Cell 4)

- **What it does:** Markdown header for the data loading section.
- **Expected Output:** Formatted text.

### 5. Define Data Directory (Cell 5)

- **What it does:** Sets the path to the data folder where your CSV files are stored. It checks if the path exists.
- **Expected Output:** Prints the absolute path to the data directory (e.g., `/home/jovyan/work/data`).

### 6. Pandas Inspection Header (Cell 6)

- **What it does:** Markdown header indicating we will first inspect data using Pandas.
- **Expected Output:** Formatted text.

### 7. Load Data into Pandas (Cell 7)

- **What it does:** Searches for `alzheimer_data_2.csv` in the data folder. If found, it reads it into a Pandas DataFrame (`df_pd`) and displays the first 5 rows.
- **Expected Output:**
  - Prints the file path being loaded.
  - Prints the data shape (rows, columns).
  - Displays a table with the first 5 rows of the dataset.

### 8. Cleaning Header (Cell 8)

- **What it does:** Markdown header for the data cleaning section.
- **Expected Output:** Formatted text.

### 9. Data Cleaning & Labeling (Cell 9)

- **What it does:**
  - Identifies the target column (`Diagnosis`).
  - Drops ID columns (`PatientID`, `DoctorInCharge`) that don't help with prediction.
  - Fills any missing values with the median of that column.
- **Expected Output:**
  - Prints the label column name.
  - Prints the list of feature columns.
  - Displays a statistical summary (count, mean, std, min, max) of the features.

### 10. Train/Test Split Header (Cell 10)

- **What it does:** Markdown header for the baseline model section.
- **Expected Output:** Formatted text.

### 11. Scikit-Learn Baseline (Cell 11)

- **What it does:** Splits the data into training and testing sets (80/20 split). Trains a standard Logistic Regression model using Scikit-Learn to establish a baseline performance benchmark.
- **Expected Output:**
  - A classification report (Precision, Recall, F1-score).
  - The ROC-AUC score (a number between 0 and 1, where 1 is perfect).

### 12. Spark Data Prep Header (Cell 12)

- **What it does:** Markdown header for moving data to Spark.
- **Expected Output:** Formatted text.

### 13. Create Spark DataFrame (Cell 13)

- **What it does:** Converts the Pandas DataFrame into a Spark DataFrame. It then uses `VectorAssembler` to combine all features into a single vector column and `StandardScaler` to normalize them.
- **Expected Output:** Displays the first 5 rows of the prepared Spark DataFrame, showing the `scaledFeatures` column.

### 14. MLlib Models Header (Cell 14)

- **What it does:** Markdown header for training Spark models.
- **Expected Output:** Formatted text.

### 15. Train Spark Models (Cell 15)

- **What it does:** Trains three different Spark MLlib models:
  1.  **Logistic Regression**
  2.  **Random Forest**
  3.  **Linear SVC (Support Vector Classifier)**
      It evaluates each one using the Area Under ROC (AUC) metric.
- **Expected Output:** Prints the ROC-AUC score for each of the three models.

### 16. Tuning Header (Cell 16)

- **What it does:** Markdown header for hyperparameter tuning.
- **Expected Output:** Formatted text.

### 17. Hyperparameter Tuning (Cell 17)

- **What it does:** Uses `CrossValidator` to find the best parameters (regularization strength) for the Logistic Regression model. This helps improve accuracy.
- **Expected Output:**
  - Prints "Running Cross-Validation...".
  - Prints the best parameters found.
  - Prints the final ROC-AUC score of the optimized model on the test set.


### 19. Graph: Class Distribution (Cell 19)

- **What it does:** Creates a bar chart showing how many patients have Alzheimer's vs. how many are healthy.
- **Graph Explanation:**
  - **X-axis:** 0 (Healthy), 1 (Alzheimer's).
  - **Y-axis:** Number of patients.
  - **Colors:** Green (Healthy), Orange (Alzheimer's).
  - **Goal:** Checks if the dataset is balanced or if one class dominates.

### 20. Graph: Confusion Matrix (Cell 20)

- **What it does:** Visualizes the predictions of the best model.
- **Graph Explanation:**
  - **Grid:** Shows 4 squares: True Negatives (Correctly predicted healthy), False Positives (Wrongly predicted Alzheimer's), False Negatives (Wrongly predicted healthy), True Positives (Correctly predicted Alzheimer's).
  - **Color:** Purple intensity indicates higher numbers.
  - **Goal:** Shows exactly where the model is making mistakes.

### 21. Graph: ROC Curve (Cell 21)

- **What it does:** Plots the True Positive Rate against the False Positive Rate.
- **Graph Explanation:**
  - **Purple Line:** The model's performance.
  - **Gray Dashed Line:** Random guessing (50% accuracy).
  - **Goal:** The closer the purple line is to the top-left corner, the better the model.

### 22. Graph: Feature Importance (Cell 22)

- **What it does:** Shows which factors contribute most to the prediction.
- **Graph Explanation:**
  - **Bars:** Each bar is a feature (e.g., Age, MMSE).
  - **Teal Bars:** Positive correlation (Higher value -> Higher risk of Alzheimer's).
  - **Orange Bars:** Negative correlation (Higher value -> Lower risk).
  - **Goal:** Identifies the most critical risk factors.

### 23. Graph: Model Comparison (Cell 23)

- **What it does:** Compares the accuracy (AUC score) of all three models.
- **Graph Explanation:**
  - **Horizontal Bars:** Length represents the score.
  - **Goal:** Quickly shows which algorithm performed best.

### 24. Graph: Correlation Heatmap (Cell 24)

- **What it does:** Shows how features relate to each other and the diagnosis.
- **Graph Explanation:**
  - **Grid:** Each cell shows the correlation between two variables (from -1 to 1).
  - **Colors:** Yellow/Green is positive correlation, Purple is negative.
  - **Goal:** Helps spot redundant features or strong predictors.

