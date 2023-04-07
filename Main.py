import time
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
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
from xgboost import XGBClassifier
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

dataset = pd.read_csv('fake reviews dataset.csv', names=['category', 'rating', 'label', 'text'])
dataset.head()

dataset['label'].value_counts()

dataset['text'] = dataset['text'].str.replace('\n', ' ')
dataset['target'] = np.where(dataset['label']=='CG', 1, 0)
dataset['target'].value_counts()

def punctuation_to_features(dataset, column):
    #punctuation is replaces with a text representation of itself
    
    dataset[column] = dataset[column].replace('!', ' exclamation ')
    dataset[column] = dataset[column].replace('?', ' question ')
    dataset[column] = dataset[column].replace('\'', ' quotation ')
    dataset[column] = dataset[column].replace('\"', ' quotation ')
    
    return dataset[column]
dataset['text'] = punctuation_to_features(dataset, 'text')

nltk.download('punkt');

def tokenize(column):
    #Tokenizes a Pandas dataframe column and returns a list of tokens.
    
    tokens = nltk.word_tokenize(column)
    return [w for w in tokens if w.isalpha()]    

dataset['tokenized'] = dataset.apply(lambda x: tokenize(x['text']), axis=1)
dataset.head()

nltk.download('stopwords');
# [nltk_data] Downloading package stopwords to /root/nltk_data...
# [nltk_data]   Package stopwords is already up-to-date!
def remove_stopwords(tokenized_column):
    #Return a list of tokens with English stopwords removed. 

    stops = set(stopwords.words("english"))
    return [word for word in tokenized_column if not word in stops]

dataset['stopwords_removed'] = dataset.apply(lambda x: remove_stopwords(x['tokenized']), axis=1)
dataset.head()

def apply_stemming(tokenized_column):
    #Return a list of tokens with Porter stemming applied.
    
    stemmer = PorterStemmer() 
    return [stemmer.stem(word).lower() for word in tokenized_column]

dataset['porter_stemmed'] = dataset.apply(lambda x: apply_stemming(x['stopwords_removed']), axis=1)
dataset.head()

def rejoin_words(tokenized_column):
    return ( " ".join(tokenized_column))
dataset['all_text'] = dataset.apply(lambda x: rejoin_words(x['porter_stemmed']), axis=1)
dataset[['all_text']].head()

X = dataset['all_text']
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)

classifiers = {}
classifiers.update({"XGBClassifier": XGBClassifier(eval_metric='logloss', objective='binary:logistic',)})
classifiers.update({"CatBoostClassifier": CatBoostClassifier(silent=True)})
classifiers.update({"LinearSVC": LinearSVC()})
classifiers.update({"MultinomialNB": MultinomialNB()})
classifiers.update({"LGBMClassifier": LGBMClassifier()})
classifiers.update({"RandomForestClassifier": RandomForestClassifier()})
classifiers.update({"DecisionTreeClassifier": DecisionTreeClassifier()})
classifiers.update({"ExtraTreeClassifier": ExtraTreeClassifier()})
classifiers.update({"AdaBoostClassifier": AdaBoostClassifier()})
classifiers.update({"KNeighborsClassifier": KNeighborsClassifier()})
classifiers.update({"RidgeClassifier": RidgeClassifier()})
classifiers.update({"SGDClassifier": SGDClassifier()})
classifiers.update({"BaggingClassifier": BaggingClassifier()})
classifiers.update({"BernoulliNB": BernoulliNB()})


dataset_models = pd.DataFrame(columns=['model', 'run_time', 'roc_auc', 'roc_auc_std'])

for key in classifiers:
    
    start_time = time.time()
    pipeline = Pipeline([("tfidataset", TfidfVectorizer()), ("clf", classifiers[key] )])
    cv = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')

    row = {'model': key,
           'run_time': format(round((time.time() - start_time)/60,2)),
           'roc_auc': cv.mean(),
           'roc_auc_std': cv.std(),
    }
    
    dataset_models = dataset_models.append(row, ignore_index=True)
    
dataset_models = dataset_models.sort_values(by='roc_auc', ascending=False)
dataset_models

bundled_pipeline = Pipeline([("tfidataset", TfidfVectorizer()), 
                             ("clf", SGDClassifier())
                            ])
bundled_pipeline.fit(X_train, y_train)
y_pred = bundled_pipeline.predict(X_test)
accuracy_score = accuracy_score(y_test, y_pred)
precision_score = precision_score(y_test, y_pred)
recall_score = recall_score(y_test, y_pred)
roc_auc_score = roc_auc_score(y_test, y_pred)
print('Accuracy:', accuracy_score)
print('Precision:', precision_score)
print('Recall:', recall_score)
print('ROC/AUC:', roc_auc_score)