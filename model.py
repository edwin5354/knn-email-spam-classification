import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import processed csv
processed_df = pd.read_csv('./csv/processed.csv')

# Preparation of X train and y train
X = processed_df.drop('Prediction', axis = 1)
y = processed_df['Prediction']

# Split the train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Some feature scaling before training and testing section
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
# Create a KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

from sklearn import metrics

# Check the training and testing score
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

train_dict = {
    'metrics': ['Accuracy', 'Precision', 'Recall Score', 'F1 Score'],
    'Training': [
        np.round(metrics.accuracy_score(y_train, y_train_pred), 2),
        np.round(metrics.precision_score(y_train, y_train_pred, average = 'weighted'), 2),
        np.round(metrics.recall_score(y_train, y_train_pred, average = 'weighted'), 2),
        np.round(metrics.f1_score(y_train, y_train_pred, average = 'weighted'), 2)
    ],
    'Testing': [
        np.round(metrics.accuracy_score(y_test, y_test_pred), 2),
        np.round(metrics.precision_score(y_test, y_test_pred, average = 'weighted'), 2),
        np.round(metrics.recall_score(y_test, y_test_pred, average = 'weighted'), 2),
        np.round(metrics.f1_score(y_test, y_test_pred, average = 'weighted'), 2)
    ]
}

metrics_df = pd.DataFrame(train_dict)
metrics_df.to_csv('./csv/org_model_metrics.csv', index= False)

# According to the metrics_df, the difference between training and testing accuracy exceeds 5%, this may indicate the fact that 
# the model is overfitting. Try using hyperparameter tuning like GridSearchCV
from sklearn.model_selection import GridSearchCV

# Create a parameter dictionary
grid_params = { 'n_neighbors' : [3,5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}

gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)
# fit the model on the training set
g_res = gs.fit(X_train, y_train)
# find the best score
print(g_res.best_score_) # 0.86802
# best parameter and use the parameters for modelling again
print(g_res.best_params_) # best param: {'metric': 'minkowski', 'n_neighbors': 3, 'weights': 'distance'}

grid_KNN = KNeighborsClassifier(n_neighbors=3, weights= 'distance', metric = 'minkowski')
grid_KNN.fit(X_train, y_train)
gy_train_pred = grid_KNN.predict(X_train)
gy_test_pred = grid_KNN.predict(X_test)

# Check the training and testing score; repeat the same process
grid_train_dict = {
    'metrics': ['Accuracy', 'Precision', 'Recall Score', 'F1 Score'],
    'Training': [
        np.round(metrics.accuracy_score(y_train, gy_train_pred), 2),
        np.round(metrics.precision_score(y_train, gy_train_pred, average = 'weighted'), 2),
        np.round(metrics.recall_score(y_train, gy_train_pred, average = 'weighted'), 2),
        np.round(metrics.f1_score(y_train, gy_train_pred, average = 'weighted'), 2)
    ],
    'Testing': [
        np.round(metrics.accuracy_score(y_test, gy_test_pred), 2),
        np.round(metrics.precision_score(y_test, gy_test_pred, average = 'weighted'), 2),
        np.round(metrics.recall_score(y_test, gy_test_pred, average = 'weighted'), 2),
        np.round(metrics.f1_score(y_test, gy_test_pred, average = 'weighted'), 2)
    ]
}

grid_metrics_df = pd.DataFrame(grid_train_dict)
grid_metrics_df.to_csv('./csv/tuned_model_metrics.csv', index= False)
# The result shows the training accuracy is 1.0 but the difference of testing accuracy is over 10% --> High chance of overfitting as well.

# Confusion matrix for the KNN model
def confusion_matrix():
    confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
    cm_display.plot()
    plt.title('Confusion Matrix of KNN Model')
    plt.savefig('./images/confusion_matrix.png')
confusion_matrix()

# Save the model for future use
import pickle
with open('./knn_model.pkl', 'wb') as knn_file: # save the knn model without hyperparameter tuning
    pickle.dump(knn, knn_file)

with open('./scaler.pkl', 'wb') as scaler_file:
    pickle.dump(sc, scaler_file)
