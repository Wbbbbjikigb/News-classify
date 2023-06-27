from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

train = pd.read_csv(r"D:\file\NLP\课设\mine\train_set.csv", sep = "\t")
test = pd.read_csv(r"D:\file\NLP\课设\mine\test_a.csv", sep = "\t")

tfidf = TfidfVectorizer(max_features=2000).fit(train["text"].iloc[:].values)
train_tfidf = tfidf.transform(train["text"].iloc[:].values)
test_tfidf = tfidf.transform(test["text"].iloc[:].values)


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier()
clf.fit(train_tfidf, train["label"].iloc[:].values)

df = pd.DataFrame()
df["label"] = clf.predict(test_tfidf)
df.to_csv("submit.csv", index = None)


from sklearn.svm import LinearSVC

clf_svc = LinearSVC()
clf_svc.fit(train_tfidf, train["label"].iloc[:].values)

df = pd.DataFrame()
df["label"] = clf_svc.predict(test_tfidf)
df.to_csv("sub_svc.csv", index = None)

from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
clf_lr.fit(train_tfidf, train["label"].iloc[:].values)

df = pd.DataFrame()
df["label"] = clf_lr.predict(test_tfidf)
df.to_csv("sub_lr.csv", index = None)