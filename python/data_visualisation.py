import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# read org_model_metrics
org_metrics = pd.read_csv('./csv/org_model_metrics.csv')
tuned_metrics = pd.read_csv('./csv/tuned_model_metrics.csv')
email_csv = pd.read_csv('./csv/emails.csv')

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

def distribution():
    email_csv['Prediction'].value_counts().plot(kind='bar', edgecolor = 'black', color='red')
    plt.title('Forecasting Spam emails')
    plt.xlabel('Spam')
    plt.ylabel('Count')
    plt.savefig('./images/spam.png')

distribution()
