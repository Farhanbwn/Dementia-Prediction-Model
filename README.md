# 🧠 Dementia Prediction Model

This repository contains a machine learning model that predicts dementia based on clinical and MRI scan data. The model utilizes various patient attributes including cognitive test scores, brain volume measurements, and demographic information to support early diagnosis of dementia.

---

## 📁 Dataset Features

The dataset contains the following columns:

| Feature       | Description |
|---------------|-------------|
| `Visit`       | Clinical visit number |
| `MR Delay`    | Days between the clinical visit and MRI scan |
| `M/F`         | Gender of the patient (M = Male, F = Female) |
| `Hand`        | Dominant hand (R = Right, L = Left) |
| `Age`         | Age of the patient in years |
| `EDUC`        | Years of formal education completed |
| `SES`         | Socioeconomic status  |
| `MMSE`        | Mini-Mental State Examination score  |
| `CDR`         | Clinical Dementia Rating |
| `eTIV`        | Estimated Total Intracranial Volume |
| `nWBV`        | Normalized Whole Brain Volume |
| `ASF`         | Atlas Scaling Factor |

---

## ⚙️ Technologies Used

- **Python 3**
- **NumPy** and **Pandas** for data manipulation
- **Scikit-learn** for building the machine learning models

---

## 🧪 Machine Learning Model

- ✅ **Support Vector Machine (SVM)**: Used for classification to predict whether a subject has dementia or not.

Model evaluation includes:
- Accuracy score
- Confusion matrix
- Classification report

---

## 📊 Objective

- Predict dementia status (binary classification)
- Perform feature engineering and preprocessing
- Visualize the relationships between cognitive, structural, and demographic features
- Assist in early-stage detection of dementia

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Farhanbwn/Dementia-Prediction-Model.git
   cd Dementia-Prediction-Model
