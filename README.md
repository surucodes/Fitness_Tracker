

## 🏋️‍♂️ Smart Fitness Tracker

**Exercise Recognition and Repetition Counting using Time Series Sensor Data**


---

### 📌 Project Overview

This project implements a real-time, ML-powered fitness tracking system that classifies physical exercises and counts repetitions using time series data from wearable sensors (accelerometer and gyroscope). It mimics and extends the capabilities of commercial fitness trackers, providing a fully custom-built ML pipeline — from raw signal processing to deployment.

---

### 🔧 Key Features

* 📉 **Sensor Signal Preprocessing** with noise filtering and outlier removal
* 🧠 **Multi-model Classification Pipeline** for exercise recognition
* 🔄 **Repetition Counting Logic** based on peak detection from signal patterns
* 📊 **Interactive Visualizations** using Plotly
* 🌐 **Deployed Streamlit App** for real-time feedback

---

### 🔬 Methodology

#### 🧹 1. Data Preprocessing

* Noise reduction using **Butterworth Low-Pass Filter**
* Outlier detection techniques:

  * **Interquartile Range (IQR)**
  * **Chauvenet’s Criterion**
  * **Local Outlier Factor (LOF)**

#### ⚙️ 2. Feature Engineering

* **Principal Component Analysis (PCA)** for dimensionality reduction
* **Temporal Abstraction** (rolling mean & std)
* **Fourier Transform** for frequency-based insights
* **KMeans Clustering** to detect latent structure

#### 🧪 3. Model Training

* Algorithms: **XGBoost, CatBoost, Random Forest, Neural Networks, Naive Bayes**
* **10-fold Cross Validation** and **Hyperparameter Tuning (RandomizedSearchCV)**
* Feature selection using **Wrapper Method with Decision Trees**

#### 🚀 4. Deployment

* Best-performing model integrated into a **Streamlit App**
* Real-time classification and repetition counting interface
* Hosted for live interaction and demo

---

### 🧰 Tech Stack

* Python (NumPy, Pandas, scikit-learn, XGBoost, CatBoost, TensorFlow)
* Streamlit
* Plotly, Matplotlib
* Jupyter Notebook

---

### 📁 Project Structure

```bash
├── data/                   # Raw and cleaned sensor data
├── notebooks/              # EDA, preprocessing, modeling experiments
├── models/                 # Saved models and evaluation results
├── scripts/                # Modular pipeline code
├── app/                    # Streamlit app interface
└── README.md
```

---

### 📈 Results

* Achieved \[99.678%] accuracy on test set for exercise classification
* Real-time repetition counting aligned with actual counts in >90% of test cases
* Robust performance across noisy and real-world-like sensor inputs

---

### 🧠 Learnings

* Deepened understanding of signal processing and noise handling
* Experimented with both deterministic and probabilistic classification models
* Built a complete ML product from data to deployment

---


