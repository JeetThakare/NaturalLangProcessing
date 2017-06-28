from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pprint
tokenize = lambda doc: doc.lower().split(" ")

document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

sklearn_tfidf = TfidfVectorizer(min_df=0, use_idf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)

query = "Putin is working to fix economic situation"


query_tfidf = sklearn_tfidf.fit_transform([query])

for _, q in enumerate(sklearn_representation.toarray()):
    print(cosine_similarity(q,query_tfidf))
    print('-------------------') #Not Working
