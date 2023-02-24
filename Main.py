import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

pd.set_option('max_colwidth', None)

df = pd.read_csv('fake reviews dataset.csv', names=['category', 'rating', 'label', 'text'])
df.head()

df['label'].value_counts()

df['text'] = df['text'].str.replace('\n', ' ')
df['target'] = np.where(df['label']=='CG', 1, 0)
df['target'].value_counts()

def punctuation_to_features(df, column):
    """Identify punctuation within a column and convert to a text representation.
    
    Args:
        df (object): Pandas dataframe.
        column (string): Name of column containing text. 
        
    Returns:
        df[column]: Original column with punctuation converted to text, 
                    i.e. "Wow! > "Wow exclamation"
    
    """
    
    df[column] = df[column].replace('!', ' exclamation ')
    df[column] = df[column].replace('?', ' question ')
    df[column] = df[column].replace('\'', ' quotation ')
    df[column] = df[column].replace('\"', ' quotation ')
    
    return df[column]
df['text'] = punctuation_to_features(df, 'text')
