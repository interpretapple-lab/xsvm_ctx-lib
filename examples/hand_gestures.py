import os
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

_path_to_lib_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path_to_lib_ not in sys.path:
    sys.path.insert(0, _path_to_lib_)
else:
    _path_to_lib_ = None

from xSVM_ctx.utils import contextualized_evaluation_process, decontextualize
from xSVM_ctx.parallel_xsvm import contextualized_xSVMC
from xSVM_ctx.explanations import show_line_chart_comparison, show_kde_comparison, show_heatmap_difference, show_line_chart_difference

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
sns.heatmap(cmn, annot=True, fmt='.3f', xticklabels=range(1, 8), yticklabels=range(1, 8))

idx_obj = 20
context_obj = 1
obj = contextualized_X_test[context_obj][idx_obj]
ev = y_pred[context_obj][idx_obj]

clf_ctx, prediction, idx_proMISV, idx_conMISV = contextualized_evaluation_process(obj, clf, context_obj)

misvPro = contextualized_X_train[0][idx_proMISV]
misvCon =  contextualized_X_train[0][idx_conMISV]
columns = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8']

classes_ref = {
    0: 'unmarked data',
    1: 'hand at rest', 
    2: 'hand clenched in a fist', 
    3: 'wrist flexion',
    4: 'wrist extension',
    5: 'radial deviations',
    6: 'ulnar deviations',
    7: 'extended palm'
}

show_line_chart_comparison(prediction, obj, columns, misvPro, misvCon, context_obj, idx_obj, output_dir=output_dir, classes_map=classes_ref)

show_kde_comparison(prediction, obj, misvPro, misvCon, context_obj, idx_obj, output_dir=output_dir, classes_map=classes_ref)

show_heatmap_difference(prediction, obj, columns, misvPro, misvCon, context_obj, idx_obj, output_dir=output_dir, classes_map=classes_ref)

show_line_chart_difference(prediction, obj, columns, misvPro, misvCon, context_obj, idx_obj, output_dir=output_dir, classes_map=classes_ref)
