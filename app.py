import streamlit as st
import pandas as pd

# All dataframes
email_df = pd.read_csv('./csv/emails_simplify.csv')

st.title('Email Spam Classification')

st.write('This exercise focuses on a supervised machine learning model that utilizes the KNN algorithm to classify emails as spam or not. We will thoroughly discuss the steps of data exploration, preprocessing, model construction, evaluation, and hyperparameter tuning.')

st.subheader('a) Data Exploration')
st.write("To begin, the dataset's structure and attributes were investigated using a variety of exploratory techniques.")

st.dataframe(email_df.head())

st.write('The dataset consists of 3,002 columns and 5,172 rows, with all the data types represented as integers.')

exp_code = '''
email_df.shape
email_df.describe()
email_df.info()
email_df.columns
'''
st.code(exp_code, language="python")

st.image('images/spam.png')
st.write("With such a high number of columns, simpler exploratory techniques are indeed more effective. In this case, analyzing the distribution of spam and non-spam emails is a great starting point. The bar plot illustrates that the dataset contains more than 3,500 non-spam emails (labeled as 0) and approximately 1,500 spam emails (labeled as 1). Exploratory techniques such as the correlation matrix are not ideal given the data's complexity.")

barplot_code = '''
def distribution():
    email_csv['Prediction'].value_counts().plot(kind='bar', edgecolor = 'black', color='red')
    plt.title('Forecasting Spam emails')
    plt.xlabel('Spam')
    plt.ylabel('Count')
    plt.savefig('./images/spam.png')

distribution()
'''
st.code(barplot_code, language="python")

st.subheader('b) Data Preprocessing')
st.write('A KNN model can be a good choice for the dataset, especially since the dataset features integer features and no categorical or binary variables. Fortunately, there are no null or missing values, so additional data cleaning is not required to perform. However, to reduce data dimensionality, a Natural Language Toolkit is utilized to filter the dataframe by removing any columns that contain words that match the stopwords.')

nlp_code = '''
import nltk
from nltk.corpus import stopwords

# Generate a list of stopwords and filter the DataFrame by removing any columns that contain words matching the stopwords.
stopwords = stopwords.words('english') # list

# Remove stopwords to compress the data a bit
for col in df.columns:
    if col in stopwords:
        df = df.drop(col, axis = 1)

df.to_csv('./csv/processed.csv', index = False)
'''
st.code(nlp_code, language="python")

st.write('Before normalizing the integer features, it is essential to separate the training and testing data.')
train_code = '''
# Preparation of X train and y train
X = processed_df.drop('Prediction', axis = 1)
y = processed_df['Prediction']

# Split the train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
st.code(train_code, language="python")

st.write('After splitting the data, scaling techniques were applied to reduce the values, ensuring that each feature contributes equally to the distance calculations. It is a good idea to normalize or standardize the integer features, as KNN is sensitive to data scaling.')

scale_code = '''
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
st.code(scale_code, language="python")

st.subheader('c) KNN Model Construction')
st.write('After scaling the data, the model is trained and tested, as demonstrated in the following code.')

model_code = '''
from sklearn.neighbors import KNeighborsClassifier
# Create a KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
'''
st.code(model_code, language="python")

st.subheader('d) Model Evaluation')

st.write('Metrics such as accuracy, precision, recall, and F1-score are assessed for both the training and testing models. Additionally, confusion matrices are used to evaluate model performance. Furthermore, the metrics scores will be saved as a CSV file to facilitate plotting the metrics as a bar chart.')

result_code = '''
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
'''

st.code(result_code, language= "python")
st.write('In this section, a bar plot is created to illustrate the metrics (accuracy, precision, recall, and F1 score) for both training and testing data. The difference between the scores exceeds 5%, indicating that the model is slightly overfitted. You can find the metric scores in the file named org_model_metrics.csv.')

st.image('./images/org_metrics.png')
acc_code = '''
def barplot(df):
    # Set the bar width and positions
    bar_width = 0.35
    index = np.arange(len(df['metrics']))
    
    # Create bars for Training and Testing
    plt.bar(index, df['Training'], bar_width, label='Training', alpha=0.7, edgecolor = 'black')
    plt.bar(index + bar_width, df['Testing'], bar_width, label='Testing', alpha=0.7, edgecolor = 'black')

    # Adding labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('KNN Model Classification Training vs Testing Metrics After Tuning')
    plt.xticks(index + bar_width / 2, df['metrics'])  # Center the tick labels
    plt.legend()  # Show the legend

    # Show the plot
    plt.tight_layout()
    plt.ylim([0.8, 1])
    plt.savefig('images/org_metrics.png')

barplot(org_metrics)
'''
st.code(acc_code, language= "python")

st.write('Based on the metrics mentioned above, a confusion matrix is plotted to evaluate the performance of the classification algorithm. This visualization helps in understanding the true positives and true negatives, providing insights into how well the model is predicting each class.')
st.image("./images/confusion_matrix.png")

confu_code = '''
# Confusion matrix for the KNN model
def confusion_matrix():
    confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
    cm_display.plot()
    plt.title('Confusion Matrix of KNN Model')
    plt.savefig('./images/confusion_matrix.png')
confusion_matrix()
'''
st.code(confu_code, language="python")

st.subheader('e) Hyperparameter Tuning')
st.write("Let’s proceed with hyperparameter tuning using techniques such as grid search or random search to enhance model performance. The optimal parameters for the KNN model are {'metric': 'minkowski', 'n_neighbors': 3, 'weights': 'distance'}. Now, let's repeat the training and testing process with these tuned parameters. Below is a bar plot that illustrates the performance metrics of the optimized KNN model.")

st.image('./images/tuned_metrics.png')

tune_code = '''
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
'''
st.code(tune_code, language='python')

st.write('Following the hyperparameter tuning, we observe a substantial increase in training accuracy, while testing accuracy remains relatively stable. This typically suggests that the model is improving its fit to the training data, which may raise concerns about overfitting.')
st.write('Further investigation into hyperparameter tuning to enhance testing accuracy is essential. Additionally, exploring the deployment of the model could be a promising direction for future research. In any case, enjoy delving into the data and the modeling process!')
