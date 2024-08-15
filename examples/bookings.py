import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns

_path_to_lib_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path_to_lib_ not in sys.path:
    sys.path.insert(0, _path_to_lib_)
else:
    _path_to_lib_ = None
from xSVM_ctx.parallel_xsvm import contextualized_xSVMC
from xSVM_ctx.utils import contextualized_evaluation_process, decontextualize
from xSVM_ctx.explanations import show_line_chart_comparison, show_kde_comparison, show_heatmap_difference, show_line_chart_difference

if _path_to_lib_:
    sys.path.remove(_path_to_lib_)
del _path_to_lib_

df1 = pd.read_csv('data/Hotel_bookings/H1.csv', )
df2 = pd.read_csv('data/Hotel_bookings/H2.csv', )
output_dir = './output/'

dropedCols = ['ArrivalDateYear', 'ArrivalDateMonth', 'ArrivalDateWeekNumber', 'ArrivalDateDayOfMonth', 'Meal', 'Country', 
              'ReservedRoomType', 'AssignedRoomType', 'DepositType', 'Agent', 'Company', 'DaysInWaitingList', 
              'ReservationStatus', 'ReservationStatusDate']

df1.drop(columns=dropedCols, inplace=True)
df2.drop(columns=dropedCols, inplace=True)

df1.dropna(axis=0, how='any', inplace=True)
df2.dropna(axis=0, how='any', inplace=True)

dfTot = pd.concat([df1, df2])

#One-hot encoding for categorical features
encoder = OneHotEncoder(sparse_output=False)
categorical_columns = ['MarketSegment', 'DistributionChannel', 'CustomerType']
encoded_features = encoder.fit_transform(dfTot[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
dfTot = dfTot.drop(columns=categorical_columns).reset_index(drop=True)
dfTot = pd.concat([dfTot, encoded_df], axis=1)

y = dfTot['IsCanceled']
X = dfTot.drop(columns=['IsCanceled'])

X_train, X_test, y_train, y_test = train_test_split(X[:len(df1)], y[:len(df1)], test_size=0.3, random_state=418, stratify=y[:len(df1)],)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X[len(df1):], y[len(df1):], test_size=0.3, random_state=418, stratify=y[len(df1):])

#Scale features to [0,1]
scaler = MinMaxScaler()
X_TrainTot = pd.concat([X_train,X_train2])
scaler.fit(X_TrainTot)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train2 = scaler.transform(X_train2)
X_test2 = scaler.transform(X_test2)

contextualized_X_train = {"H1": X_train, "H2": X_train2}
contextualized_y_train = {"H1": y_train, "H2": y_train2}

contextualized_X_test = {"H1": X_test, "H2": X_test2}
contextualized_y_test = {"H1": y_test, "H2": y_test2}

w_0 = 44220/(44220 + 75166)
w_1 = 1 - w_0
weights = {0: w_0, 1: w_1}

clf = contextualized_xSVMC(kernel='rbf', C=100, gamma=3, class_weight=weights)

clf.fit(contextualized_X_train, contextualized_y_train, n_jobs=2)

y_pred = clf.predict_with_context_by_voting(contextualized_X_test, n_jobs=2, verbose=50)

test_list = decontextualize(contextualized_y_test, clf.contexts_)
pred_list = decontextualize(y_pred, clf.contexts_)

print(f"Accuracy: {accuracy_score(test_list, pred_list)}")
print(f"F1 Score: {f1_score(test_list, pred_list, average='weighted')}")
print(f"Precision: {precision_score(test_list, pred_list, average='weighted')}")
print(f"Recall: {recall_score(test_list, pred_list, average='weighted')}")
print(classification_report(test_list, pred_list))

cm = confusion_matrix(test_list, pred_list)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cmn, annot=True, fmt='.3f')

idx_obj = 48
context_obj = "H1"
obj = contextualized_X_test[context_obj][idx_obj]
ev = y_pred[context_obj][idx_obj]

clf_ctx, prediction, idx_proMISV, idx_conMISV = contextualized_evaluation_process(obj, clf, context_obj)
misvPro = X_train[idx_proMISV] # X_train is in context H1
misvCon = X_train[idx_conMISV]
columns = X.columns

classes_ref = {0: "Not Canceled", 1: "Canceled"}

show_line_chart_comparison(prediction, obj, columns, misvPro, misvCon, context_obj, idx_obj, output_dir=output_dir, classes_map=classes_ref, explanation_type="complete")

show_kde_comparison(prediction, obj, misvPro, misvCon, context_obj, idx_obj, output_dir=output_dir, classes_map=classes_ref, explanation_type="complete")

show_heatmap_difference(prediction, obj, columns, misvPro, misvCon, context_obj, idx_obj, output_dir=output_dir, classes_map=classes_ref, explanation_type="complete")

show_line_chart_difference(prediction, obj, columns, misvPro, misvCon, context_obj, idx_obj, output_dir=output_dir, classes_map=classes_ref, explanation_type="complete")
