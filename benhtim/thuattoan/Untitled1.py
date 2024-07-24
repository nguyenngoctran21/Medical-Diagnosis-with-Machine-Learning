# Regular EDA and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import argparse
import sys

# Models from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scikitplot.metrics import plot_confusion_matrix 
# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from scikitplot.metrics import plot_roc_curve
# Function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_scores = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        model_scores[model_name] = score
        
    return model_scores
# Loading the dataset
df = pd.read_csv('data/heart-disease.csv')

# Exploratory Data Analysis (EDA)
# ... (Your EDA code)

# Splitting the data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'LogisticRegression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVM': SVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()  # Thêm mô hình GradientBoostingClassifier
}

# Fit and score models
model_scores = fit_and_score(models, X_train, X_test, y_train, y_test)

# Model comparison
model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot(kind='bar');
import argparse

# Tạo đối tượng ArgumentParser
parser = argparse.ArgumentParser(description='Your program description')

# Thêm các tham số dòng lệnh
parser.add_argument('--algorithm', type=str, help='Specify the algorithm')
parser.add_argument('--C', type=float, help='Specify the value of C')
parser.add_argument('--solver', type=str, help='Specify the solver')

# Phân tích các tham số từ dòng lệnh
args = parser.parse_args()

# Truy cập giá trị của C và solver từ tham số dòng lệnh
C_value = args.C
solver_value = args.solver


# In giá trị của C và solver
print(f'Giá trị C nhận từ dòng lệnh: {C_value}')
print(f'Giá trị solver nhận từ dòng lệnh: {solver_value}')

# Hyperparameter tuning for Logistic Regression
log_reg_grid = {
    'C': [C_value],
    'solver': [solver_value],
   
}


gs_log_reg = GridSearchCV(LogisticRegression(), log_reg_grid, cv=2, verbose=True)
gs_log_reg.fit(X_train, y_train)

# Evaluate the tuned model
print(gs_log_reg.best_params_['C'])
print(gs_log_reg.score(X_test, y_test))

# Confusion matrix and classification report
plot_confusion_matrix(y_test, gs_log_reg.predict(X_test), title='Confusion Matrix')
print(classification_report(y_test, gs_log_reg.predict(X_test)))

# Cross-validated metrics
cv_accuracy = np.mean(cross_val_score(gs_log_reg, X, y, scoring='accuracy', cv=5))
cv_precision = np.mean(cross_val_score(gs_log_reg, X, y, scoring='precision', cv=5))
cv_recall = np.mean(cross_val_score(gs_log_reg, X, y, scoring='recall', cv=5))
cv_f1 = np.mean(cross_val_score(gs_log_reg, X, y, scoring='f1', cv=5))
cv_auc_roc = np.mean(cross_val_score(gs_log_reg, X, y, scoring='roc_auc', cv=5))

# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({'Accuracy': cv_accuracy,
                            'Precision': cv_precision,
                            'Recall': cv_recall,
                            'F1': cv_f1,
                            'AUC-ROC': cv_auc_roc},
                              index=[0])
cv_metrics.T.plot.bar(legend=False);
plt.title('Cross Validated Classification Metrics');
plt.xticks(rotation=30);

# Feature importance visualization
feature_dict = dict(zip(df.columns, list(gs_log_reg.best_estimator_.coef_[0])))
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title='Feature Importance of Logistic Regression', legend=False);

# Prediction function
def predict_Heart_Disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    x = np.zeros(len(X.columns))
    x[0] = age
    x[1] = sex
    x[2] = cp
    x[3] = trestbps
    x[4] = chol
    x[5] = fbs
    x[6] = restecg
    x[7] = thalach
    x[8] = exang
    x[9] = oldpeak
    x[10] = slope
    x[11] = ca
    x[12] = thal
    return gs_log_reg.predict([x])[0]

import os
import datetime
import argparse

# Tạo thư mục nếu nó chưa tồn tại
result_directory = 'results'
os.makedirs(result_directory, exist_ok=True)



# Lấy thông tin về thời gian hiện tại
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Tạo tên tệp tin có chứa thông tin về thời gian chạy, siêu tham số và đặt trong thư mục "results"
output_file_name = f'model_{args.algorithm}_C_{args.C}_solver_{args.solver}_{current_time}.txt'
output_file_path = os.path.join(result_directory, output_file_name)

# Mở tệp tin để ghi kết quả
with open(output_file_path, 'w') as file:
    # Ghi kết quả vào tệp tin
    file.write("Kết quả của mô hình:\n")
    file.write(f"Siêu tham số C: {args.C}\n")
    file.write(f"Siêu tham số solver: {args.solver}\n")
    file.write(f"Tên giải thuật: {args.algorithm}\n")
    file.write("...")  # Ghi kết quả thực tế ở đây
    # Ghi thông tin về các mô hình đã đánh giá
    file.write("Model Scores:\n")
    for model_name, score in model_scores.items():
        file.write(f"{model_name}: {score:.4f}\n")

    # Ghi thông tin về mô hình Logistic Regression sau tinh chỉnh
    file.write("\nLogistic Regression Best Parameters:\n")
    file.write(f"C: {gs_log_reg.best_params_['C']}\n")
    file.write(f"Test Accuracy: {gs_log_reg.score(X_test, y_test):.4f}\n")

    # Ghi thông tin về confusion matrix và classification report
    file.write("\nConfusion Matrix:\n")
    file.write(f"{confusion_matrix(y_test, gs_log_reg.predict(X_test))}\n\n")

    file.write("\nClassification Report:\n")
    file.write(f"{classification_report(y_test, gs_log_reg.predict(X_test))}\n")

    # Ghi thông tin về các metrics cross-validated
    file.write("\nCross-validated Metrics:\n")
    file.write(f"Accuracy: {cv_accuracy:.4f}\n")
    file.write(f"Precision: {cv_precision:.4f}\n")
    file.write(f"Recall: {cv_recall:.4f}\n")
    file.write(f"F1: {cv_f1:.4f}\n")
    file.write(f"AUC-ROC: {cv_auc_roc:.4f}\n")

   

