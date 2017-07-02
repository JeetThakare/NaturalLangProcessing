import csv
from textblob import TextBlob
import pandas as pd
import sklearn
import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve


# def split_into_tokens(message):
#     message = unicode(message, 'utf8')
#     return TextBlob(message).words


def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]


messages = pd.read_csv('./smsspamcollection/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,
                       names=["label", "message"])
# print messages

# messages['length'] = messages['message'].map(lambda text: len(text))

# messages = pd.DataFrame(messages.message.apply(split_into_lemmas))

vectorizer = CountVectorizer(analyzer=split_into_lemmas)
messages_Vect = vectorizer.fit_transform(messages.message)


tfidf_transformer = TfidfTransformer()
messages_tfidf = tfidf_transformer.fit_transform(messages_Vect)


# print(messages.head())

# print tfidf_transformer.idf_[vectorizer.vocabulary_['hello']]

spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])

pipeline = Pipeline([
    ('vec', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
# print scores

print(spam_detector.predict("hi, how are you"))
