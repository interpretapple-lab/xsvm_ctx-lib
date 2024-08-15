import os
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
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

base_path = 'data/EMG_data_for_hand_gestures/'
output_dir = './output/'

data = []

for i in range(1, 37):
    folder_path = os.path.join(base_path, f'{i:02}')
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    df1 = pd.read_csv(os.path.join(folder_path, files[0]), sep='\t')
    df2 = pd.read_csv(os.path.join(folder_path, files[1]), sep='\t')
    combined_df = pd.concat([df1, df2])
    
    combined_df['context'] = i
    combined_df.dropna(axis=0, how='any', inplace=True)
    combined_df = combined_df[combined_df['class'] != 0]
    
    channels = combined_df[['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8']].values
    channels_normalized = preprocessing.normalize(channels, norm='l2')
    
    context_data = list(zip(channels_normalized, combined_df['class']))
    data.append(context_data)

tot = 0
train_data = {}
contextualized_X_train = {}
contextualized_y_train = {}
contextualized_X_test = {}
contextualized_y_test = {}
for i in range(0, len(data)):
    X = np.array([item[0] for item in data[i]])
    y = np.array([item[1] for item in data[i]])
    tot += len(data[i])        
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=418, stratify=y)
        
    contextualized_X_train[i] = X_train
    contextualized_y_train[i] = y_train

    contextualized_X_test[i] = X_test
    contextualized_y_test[i] = y_test

clf = contextualized_xSVMC(kernel='rbf', C=100, gamma=20) 

clf.fit(contextualized_X_train, contextualized_y_train)


selected_X, selected_y, remaining_X, remaining_y = select_n_instances(contextualized_X_test[1], contextualized_y_test[1], 3)

y_pred = clf.predict_with_unknown_context(remaining_X, selected_X, selected_y)

pred_list = [topK[0].class_name for topK in y_pred]

print(f"Accuracy: {accuracy_score(remaining_y, pred_list)}")
print(f"F1 Score: {f1_score(remaining_y, pred_list, average='weighted')}")
print(f"Precision: {precision_score(remaining_y, pred_list, average='weighted')}")
print(f"Recall: {recall_score(remaining_y, pred_list, average='weighted')}")
print(classification_report(remaining_y, pred_list))
