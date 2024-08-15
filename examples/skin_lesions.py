import os
import sys
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
import numpy as np

_path_to_lib_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path_to_lib_ not in sys.path:
    sys.path.insert(0, _path_to_lib_)
else:
    _path_to_lib_ = None

from xSVM_ctx.parallel_xsvm import contextualized_xSVMC
from xSVM_ctx.utils import load_images_context_and_labels, train_test_split_with_context, group_data_by_context, decontextualize, contextualized_evaluation_process
from xSVM_ctx.explanations import show_object_comparison, show_influence_map

if _path_to_lib_:
    sys.path.remove(_path_to_lib_)
del _path_to_lib_

directory = 'data/Skin_conditions/Images/'
metadata_file = 'data/Skin_conditions/processed_metadata.csv'
output_dir = './output/'

images, labels, context = load_images_context_and_labels(directory, metadata_file, context_column="localization")

X_train, y_train, context_train, X_test, y_test, context_test = train_test_split_with_context(images, labels, context)

flattened_X_train = [image.flatten() for image in X_train]
flattened_X_test = [image.flatten() for image in X_test]

X_train_normalized = preprocessing.normalize(flattened_X_train, norm='l2')
X_test_normalized = preprocessing.normalize(flattened_X_test, norm='l2')

contextualized_X_train = group_data_by_context(X_train_normalized, context_train)
contextualized_y_train = group_data_by_context(y_train, context_train)

clf = contextualized_xSVMC(kernel='rbf', C=100, gamma=20, class_weight='balanced')

clf.fit(contextualized_X_train, contextualized_y_train, verbose=50, n_jobs=6)

contextualized_data_test = group_data_by_context(X_test_normalized, context_test)
y_pred = clf.predict_with_context_by_voting(contextualized_data_test)
contextualized_y_test = group_data_by_context(y_test, context_test)

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

idx_obj = 15
context_obj = "torso"
obj = contextualized_data_test[context_obj][idx_obj]
ev = y_pred[context_obj][idx_obj]

clf_ctx, prediction, idx_proMISV, idx_conMISV = contextualized_evaluation_process(obj, clf, context_obj)

contextualized_X_train = group_data_by_context(X_train, context_train)

img = contextualized_X_train[context_obj][idx_obj]
misvPro = contextualized_X_train[context_obj][idx_proMISV]
misvCon = contextualized_X_train[context_obj][idx_conMISV]

show_object_comparison(prediction, context_obj, idx_obj, img, misvPro, misvCon, output_dir)

contextualized_X_test = group_data_by_context(flattened_X_test, context_test)
y_train_context = contextualized_y_train[context_obj]
class_pro = y_train_context[idx_proMISV]
class_con = y_train_context[idx_conMISV]
imgV = contextualized_X_test[context_obj][idx_obj]

show_influence_map(clf_ctx, prediction, img, imgV, misvPro, misvCon, idx_obj, class_pro, class_con, output_dir)

