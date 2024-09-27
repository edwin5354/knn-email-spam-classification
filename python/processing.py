import pandas as pd
import nltk
from nltk.corpus import stopwords

df = pd.read_csv('./csv/emails.csv')

# Remove columns with object as data type
df = df.drop('Email No.', axis=1)

# Generate a list of stopwords and filter the DataFrame by removing any columns that contain words matching the stopwords.
stopwords = stopwords.words('english') # list

# Remove stopwords to compress the data a bit
for col in df.columns:
    if col in stopwords:
        df = df.drop(col, axis = 1)

df.to_csv('./csv/processed.csv', index = False)
