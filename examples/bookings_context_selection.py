import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

_path_to_lib_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path_to_lib_ not in sys.path:
    sys.path.insert(0, _path_to_lib_)
else:
    _path_to_lib_ = None

from xSVM_ctx.parallel_xsvm import contextualized_xSVMC
from xSVM_ctx.utils import select_n_instances

if _path_to_lib_:
    sys.path.remove(_path_to_lib_)
del _path_to_lib_


df1 = pd.read_csv('data/Hotel_bookings/H1.csv', )
df2 = pd.read_csv('data/Hotel_bookings/H2.csv', )
output_dir = './output/'
df1
#quitar columns no requeridas
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

clf = contextualized_xSVMC(kernel='rbf', C=100, gamma=3, decision_function_shape="ovr")

clf.fit(contextualized_X_train, contextualized_y_train, n_jobs=2)

selected_X, selected_y, remaining_X, remaining_y = select_n_instances(contextualized_X_test["H2"], contextualized_y_test["H2"], 10)
y_pred = clf.predict_with_unknown_context(remaining_X, selected_X, selected_y)

pred_list = [topK[0].class_name for topK in y_pred]

print(f"Accuracy: {accuracy_score(remaining_y, pred_list)}")
print(f"F1 Score: {f1_score(remaining_y, pred_list, average='weighted')}")
print(f"Precision: {precision_score(remaining_y, pred_list, average='weighted')}")
print(f"Recall: {recall_score(remaining_y, pred_list, average='weighted')}")
print(classification_report(remaining_y, pred_list))