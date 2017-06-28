from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

que1 = "how can I go form Pune to Mumbai"
que2 = "How is the weather in Pune today"

ans1 = "Get a cab"
ans2 = "Check Google Now"

vect = TfidfVectorizer(stop_words='english', min_df=1)
