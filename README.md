# AI Engineering Portfolio
Documenting my 8-month journey to becoming an AI Engineer. 

## Phase 0: Python for AI & Data Handling
**File:** `Phase_0_Titanic_Data_Analysis.ipynb`
**Description:** This notebook demonstrates foundational data manipulation using pandas and matplotlib. 
**Skills Demonstrated:**
*   Vectorized data cleaning (handling NaN values via imputation and dropping).
*   Data aggregation using `groupby`.
*   Data visualization (bar charts and histograms) to extract insights on passenger survival rates.

## Phase 1: Classical Machine Learning & Evaluation

### Project 1: Supervised Learning (Naval Strength)
**File:** `Phase_1_Naval_Strength_Prediction.ipynb`
**Description:** Built and evaluated predictive models using a geopolitical naval fleet dataset to analyze strategic military capabilities.

**Skills Demonstrated:**
*   **Data Splitting:** Applied the Train/Test split methodology to prevent data leakage and model overfitting.
*   **Regression:** Trained a Linear Regression model to predict a continuous overall Strength Index and extracted feature coefficients to analyze the mathematical weight of aircraft carriers versus submarines.
*   **Classification:** Implemented a Logistic Regression model to categorize fleets as Strategic Power Projection vs. Regional Defense.
*   **Model Evaluation:** Evaluated models using Root Mean Squared Error (RMSE) for regression, alongside Accuracy, Precision, and Recall metrics for classification.

### Project 2: Unsupervised Learning (Customer Segmentation)
**File:** `Phase_1_Retail_Customer_Clustering.ipynb`
**Description:** Transitioned from clean data to messy, real-world retail data to mathematically discover hidden buyer personas using unsupervised learning.

**Skills Demonstrated:**
*   **Data Cleaning:** Handled missing database entries using median imputation to ensure structural integrity for distance-based algorithms.
*   **Feature Scaling:** Applied `StandardScaler` to normalize dimensions and prevent magnitude bias in distance calculations.
*   **Dimensionality Reduction:** Utilized Principal Component Analysis (PCA) to compress multidimensional behavioral data into a 2D space for visualization.
*   **Clustering:** Deployed K-Means Clustering to mathematically carve the customer base into distinct, actionable business personas, translating algorithmic output into strategic insights using `groupby`.

## Phase 2: Deep Learning & Neural Networks

### Project 3: Computer Vision (Handwritten Digit Recognition)

**File:** `Phase_2_MNIST_Digit_Recognition.ipynb`
**Description:** Built and trained a fully connected Artificial Neural Network (ANN) using TensorFlow and Keras to "see" and classify images of handwritten digits (MNIST dataset).

**Skills Demonstrated:**
*   **Deep Learning Architecture:** Designed a Sequential neural network utilizing Flatten and Dense layers with ReLU and Softmax activation functions.
*   **Image Preprocessing:** Normalized 2D pixel arrays (0-255) down to a 0.0-1.0 scale to optimize mathematical weight updates and prevent gradient explosion.
*   **Model Compilation & Training:** Configured the `adam` optimizer and `sparse_categorical_crossentropy` loss function to iteratively adjust 101,770 trainable parameters over multiple epochs.
*   **Evaluation & Inference:** Validated model generalization against a holdout test set (achieving ~98% accuracy) and utilized `matplotlib` to visually map the AI's real-time predictions to raw pixel data.

