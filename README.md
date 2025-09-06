# House Prices – Machine Learning Models

This project trains and compares multiple machine learning models to predict house prices using tabular data.  
It is based on the Kaggle House Prices dataset format (`train.csv` and `test.csv`) and demonstrates preprocessing, model building, and performance evaluation.

---

## Project Structure
- `train.csv` → Training dataset (with target: SalePrice)  
- `test.csv` → Test dataset (without target)  
- `mlm.py` → Main script with models and pipelines  
- `README.md` → Project documentation  

---

## Requirements
- Python 3.9+  
- Libraries: `numpy`, `pandas`, `scikit-learn`, `tensorflow` (optional for Neural Network)  

---

## Install dependencies:
- pip install numpy pandas scikit-learn tensorflow

---

The script will:  
- Load `train.csv` and `test.csv`  
- Preprocess data (imputation, scaling, encoding)  
- Train and evaluate: Linear Regression, Random Forest, Gradient Boosting, Neural Network (if TensorFlow is installed)  
- Print metrics: R², RMSE, and MAPE  
