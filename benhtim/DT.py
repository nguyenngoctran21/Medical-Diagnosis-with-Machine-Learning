# Regular EDA and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import argparse

# Models from scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from scikitplot.metrics import plot_confusion_matrix

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
    'DecisionTreeClassifier': DecisionTreeClassifier()
}

# Fit and score models
model_scores = fit_and_score(models, X_train, X_test, y_train, y_test)

# Model comparison
model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot(kind='bar')
# plt.show()

# Argument parsing
parser = argparse.ArgumentParser(description='Your program description')

#thêm tham số dòng lệnh
parser.add_argument('--algorithm', type=str, help='Specify the algorithm')
parser.add_argument('--criterion', type=str, help='Specify the criterion')
parser.add_argument('--max_depth', type=int, help='Specify the maximum depth')
parser.add_argument('--min_samples_split', type=int, help='Specify the minimum number of samples required to split an internal node')


args = parser.parse_args()

# Accessing values from command-line arguments
algorithm_value = args.algorithm
criterion_value = args.criterion
max_depth_value = args.max_depth
min_samples_split_value = args.min_samples_split

# Print the values
print(f'Giá trị algorithm nhận từ dòng lệnh: {algorithm_value}')
print(f'Criterion value received from command line: {criterion_value}')
print(f'Max depth value received from command line: {max_depth_value}')
print(f'Min samples split value received from command line: {min_samples_split_value}')

# Hyperparameter tuning for Decision Trees
dt_grid = {
    'algorithm': ['gini'] ,
    'criterion': [criterion_value],
    'max_depth': [max_depth_value],
    'min_samples_split': [min_samples_split_value],
}

gs_dt = GridSearchCV(DecisionTreeClassifier(), dt_grid, cv=2, verbose=True)
gs_dt.fit(X_train, y_train)

# Confusion matrix and classification report
plot_confusion_matrix(y_test, gs_dt.predict(X_test), title='Confusion Matrix')
print(classification_report(y_test, gs_dt.predict(X_test)))

# Cross-validated metrics
cv_accuracy = np.mean(cross_val_score(gs_dt, X, y, scoring='accuracy', cv=5))
cv_precision = np.mean(cross_val_score(gs_dt, X, y, scoring='precision', cv=5))
cv_recall = np.mean(cross_val_score(gs_dt, X, y, scoring='recall', cv=5))
cv_f1 = np.mean(cross_val_score(gs_dt, X, y, scoring='f1', cv=5))
cv_auc_roc = np.mean(cross_val_score(gs_dt, X, y, scoring='roc_auc', cv=5))

# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({'Accuracy': cv_accuracy,
                            'Precision': cv_precision,
                            'Recall': cv_recall,
                            'F1': cv_f1,
                            'AUC-ROC': cv_auc_roc},
                              index=[0])
cv_metrics.T.plot.bar(legend=False)
plt.title('Cross Validated Classification Metrics')
plt.xticks(rotation=30)
plt.show()

# Visualization of feature importance
feature_dict = dict(zip(df.columns, list(gs_dt.best_estimator_.feature_importances)))


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
    return gs_dt.predict([x])[0]


# Tạo thư mục nếu nó chưa tồn tại
result_directory = 'results'
os.makedirs(result_directory, exist_ok=True)



# Lấy thông tin về thời gian hiện tại
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Tạo tên tệp tin có chứa thông tin về thời gian chạy, siêu tham số và đặt trong thư mục "results"
output_file_name = f'model_{args.algorithm}_criterion_{args.criterion}_max_depth_{args.max_depth}_min_samples_split_{args.min_samples_split}{current_time}.txt'
output_file_path = os.path.join(result_directory, output_file_name)

# Mở tệp tin để ghi kết quả
with open(output_file_path, 'w') as file:
    # Ghi kết quả vào tệp tin
    file.write("Kết quả của mô hình với:\n")
    file.write(f"Tên giải thuật: {args.algorithm}\n")
    file.write(f"Siêu tham số weights: {args.weights}\n")
    file.write(f"Siêu tham số p: {args.p}\n")
    file.write(f"Tên giải thuật: {args.algorithm}\n")
    file.write("...")  # Ghi kết quả thực tế ở đây

    # Ghi thông tin về các mô hình đã đánh giá
    file.write("Model Scores:\n")
    for model_name, score in model_scores.items():
        file.write(f"{model_name}: {score:.4f}\n")

    # Ghi thông tin về confusion matrix và classification report
    file.write("\nConfusion Matrix:\n")#matran nham lan
    file.write(f"{confusion_matrix(y_test, gs_dt.predict(X_test))}\n\n")
    file.write("\nClassification Report:\n")# báo cáo tổng hợp
    file.write(f"{classification_report(y_test, gs_dt.predict(X_test))}\n")

    # Ghi thông tin về các metrics cross-validated
    file.write("\nCross-validated Metrics:\n")# danh gia theo folder
    file.write(f"Accuracy: {cv_accuracy:.4f}\n")
    file.write(f"Precision: {cv_precision:.4f}\n")
    file.write(f"Recall: {cv_recall:.4f}\n")
    file.write(f"F1: {cv_f1:.4f}\n")
    file.write(f"AUC-ROC: {cv_auc_roc:.4f}\n")

   

