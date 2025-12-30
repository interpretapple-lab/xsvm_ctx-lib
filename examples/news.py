import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

_path_to_lib_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path_to_lib_ not in sys.path:
    sys.path.insert(0, _path_to_lib_)
else:
    _path_to_lib_ = None
from xSVM_ctx.utils import train_test_split_with_context, group_data_by_context, decontextualize, contextualized_evaluation_process
from xSVM_ctx.parallel_xsvm import contextualized_xSVMC
from xSVM_ctx.explanations import show_hierarchical_clustering, show_word_comparison

if _path_to_lib_:
    sys.path.remove(_path_to_lib_)
del _path_to_lib_

output_dir = './output/'

col_names = ["y", "Title", "Description"]
df_train = pd.read_csv("./data/news/train.csv", names=col_names)
df_test = pd.read_csv("./data/news/test.csv", names=col_names)

def get_agnews_ctx(title, description):
    ctx = re.search(r"\(([^)]+)\)", title)
    if ctx:
        return ctx.group(1)

    ctx = re.search(r"\(([^)]+)\)", description)
    if ctx:
        return ctx.group(1)

    return "Unknown"

df_train["context"] = df_train.apply(lambda row: get_agnews_ctx(row["Title"], row["Description"]), axis=1)
df_test["context"] = df_test.apply(lambda row: get_agnews_ctx(row["Title"], row["Description"]), axis=1)

df_train_comb = df_train.copy()
df_train_comb["text"] = df_train_comb["Title"] + " " + df_train_comb["Description"]

df_test_comb = df_test.copy()
df_test_comb["text"] = df_test_comb["Title"] + " " + df_test_comb["Description"]

df_all = pd.concat([df_train_comb, df_test_comb], ignore_index=True)

all_dist = df_all.groupby(["context", "y"]).size().reset_index(name="count")
all_dist = all_dist[all_dist["count"] > 160].groupby("context").size()
all_dist = all_dist[all_dist == 4].index.tolist()

df_all_filtered = df_all[(df_all["context"].isin(all_dist)) & (df_all["context"] != "Unknown")]
X = df_all_filtered["text"].reset_index(drop=True)
y = df_all_filtered["y"].reset_index(drop=True)
context = df_all_filtered["context"].reset_index(drop=True)

X_train, y_train, context_train, X_test, y_test, context_test = train_test_split_with_context(X, y, context, test_size=0.3, random_state=418)

stop_words = stopwords.words("english") + ['company', 'year', 'yesterday', 'week', 'new', "reuters", "ap", "afp"]
stemmer = PorterStemmer()
p = re.compile(r'[a-zA-Z]+')
def tokenize(text):
    min_length = 3
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stop_words]
    tokens = [stemmer.stem(w) for w in words]
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens

vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized  = vectorizer.transform(X_test)

scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train_vectorized)
X_test_scaled  = scaler.transform(X_test_vectorized)

contextualized_X_train = group_data_by_context(X_train_scaled.toarray(), context_train)
contextualized_X_test = group_data_by_context(X_test_scaled.toarray(), context_test)
contextualized_y_train = group_data_by_context(y_train, context_train)
contextualized_y_test = group_data_by_context(y_test, context_test)

clf = contextualized_xSVMC(kernel="linear", random_state=418, class_weight="balanced")

clf.fit(contextualized_X_train, contextualized_y_train, n_jobs=3)

y_pred = clf.predict_with_context_by_voting(contextualized_X_test, n_jobs=3, verbose=50)

test_list = decontextualize(contextualized_y_test, clf.contexts_)
pred_list = decontextualize(y_pred, clf.contexts_)

print(f"Accuracy: {accuracy_score(test_list, pred_list)}")
print(f"F1 Score: {f1_score(test_list, pred_list, average='weighted')}")
print(f"Precision: {precision_score(test_list, pred_list, average='weighted')}")
print(f"Recall: {recall_score(test_list, pred_list, average='weighted')}")
print(classification_report(test_list, pred_list))

cm = confusion_matrix(test_list, pred_list)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
classes_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
labels = [classes_map[i] for i in sorted(classes_map.keys())]
sns.heatmap(cmn, annot=True, fmt='.3f', xticklabels=labels, yticklabels=labels)

idx_obj = 1
context_obj = 'AP'
obj = contextualized_X_test[context_obj][idx_obj]
ev = y_pred[context_obj][idx_obj]

clf_ctx, prediction, idx_proMISV, idx_conMISV = contextualized_evaluation_process(obj, clf, context_obj)

contextualized_text_train = group_data_by_context(X_train, context_train)
contextualized_text_test = group_data_by_context(X_test, context_test)

misvPro = contextualized_text_train[context_obj][idx_proMISV]
misvCon =  contextualized_text_train[context_obj][idx_conMISV]
evaluated_object = contextualized_text_test[context_obj][idx_obj]

show_hierarchical_clustering(prediction, context_obj, idx_obj, evaluated_object, misvPro, misvCon, vectorizer, output_dir=output_dir, classes_map=classes_map)
show_word_comparison(prediction, context_obj, idx_obj, evaluated_object, misvPro, misvCon, vectorizer, output_dir=output_dir, classes_map=classes_map)