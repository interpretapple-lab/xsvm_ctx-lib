import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import itertools


def load_images_context_and_labels(directory, metadata_file, image_column="image_id", context_column="context"):
    """ Loads images, context and labels from a directory and a metadata file.

    Parameters:
        directory: Directory where the images are stored.
        metadata_file: Path to the metadata file.
        image_column: Name of the column that contains the image ids.
        context_column: Name of the column that contains the context of the images.

    Returns:
        images: List of images.
        labels: List of labels.
        contexts: List of contexts.
    """
    metadata = pd.read_csv(metadata_file)
    image_ids = metadata[image_column]
    image_paths = [os.path.join(directory, f"{image_id}.jpg") for image_id in image_ids]
    images = [cv2.imread(path) for path in image_paths]
    valid_indices = [i for i, image in enumerate(images) if image is not None]
    images = [images[i] for i in valid_indices]
    labels = [metadata['dx'][i] for i in valid_indices]
    contexts = [metadata[context_column][i] for i in valid_indices]
    return images, labels, contexts


def train_test_split_with_context(X, y, contexts, test_size=0.2, random_state=42):
    """ Splits the data into training and test sets while keeping the context and classes distribution in both sets.
    
    Parameters:
        X: The input data of shape (n_samples, n_features).
        y: The target classes of shape (n_samples,).
        contexts: List of contexts.
        test_size: The proportion of the dataset to include in the test split.
        random_state: Controls the shuffling applied to the data before applying the split.

    Returns:
        X_train: The input data of the training set.
        y_train: The target labels of the training set.
        context_train: The contexts of the training set.
        X_test: The input data of the test set.
        y_test: The target labels of the test set.
        context_test: The contexts of the test set.
    """
    combined = list(zip(X, y, contexts))
    stratify_label_context = [f'{context} + {label}' for context, label in zip(contexts, y)]
    train_combined, test_combined = train_test_split(combined, test_size=test_size, random_state=random_state,
                                                     stratify=stratify_label_context)
    X_train, y_train, context_train = zip(*train_combined)
    X_test, y_test, context_test = zip(*test_combined)
    return X_train, y_train, context_train, X_test, y_test, context_test


def group_data_by_context(data, context):
    """ Groups data by context.

    Parameters:
        data: Data of shape (n_samples, n_features).
        context: List of contexts of shape (n_samples,).

    Returns:
        contextualized_data: Dictionary with the shape {context: X} where X is the data of shape (n_samples, n_features).
    """
    contextualized_data = {}
    for i in range(len(context)):
        if context[i] not in contextualized_data:
            contextualized_data[context[i]] = []
        contextualized_data[context[i]].append(data[i])
    return contextualized_data


def get_model_by_context(clf, context):
    """ Gets the xSVMC model trained in the given context.

    Parameters:
        clf: The parallel_xSVMC model.
        context: The selected context.

    Returns:
        xSVMC:  xSVMC model trained in the given context.
    """
    return clf.contextualized_clfs_.get(context, None)


def decontextualize(contextualized_dict, context_list):
    """ Decontextualizes a list of predictions given a list of contexts.

    Parameters:
        context_list: List of contexts of shape (n_samples,).
        contextualized_dict: Dictionary with the shape {context: y} where y is a ndarray of shape (n_samples,)
        or a ndarray of shape (n_samples, k).

    Returns:
        decontextualized_list: List of shape (n_samples,).
    """
    decontextualized_list = [contextualized_dict[context] for context in context_list]
    decontextualized_list = list(itertools.chain(*decontextualized_list))
    try:
        decontextualized_list = [topK[0].class_name for topK in decontextualized_list]
    finally:
        return decontextualized_list


def contextualized_evaluation_process(obj, contextualized_model, context):
    """ Evaluates an object in the given context.

    Parameters:
        obj: The object to evaluate.
        contextualized_model: The parallel_xSVMC model.
        context: The selected context.

    Returns:
        clf: The xSVMC model trained in the given context.
        prediction: The prediction of the object.
        idx_proMISV: The index of the positive most influential support vector in the training set.
        idx_conMISV: The index of the negative most influential support vector in the training set.
    """
    clf = get_model_by_context(contextualized_model, context)
    ref_SVs = clf.support_
    topK = clf.predict_with_context([obj])[0]
    prediction = topK[0]
    idx_proMISV = ref_SVs[prediction.eval.membership.reason]
    idx_conMISV = ref_SVs[prediction.eval.nonmembership.reason]
    return clf, prediction, idx_proMISV, idx_conMISV


def select_n_instances(X, y, n):
    """ Selects the first n instances of each class in the dataset.

    Parameters:
        X: The input data of shape (n_samples, n_features).
        y: The target classes of shape (n_samples,).
        n: Number of instances to select for each class.

    Returns:
        selected_X: The input data of the selected instances.
        selected_y: The target labels of the selected instances.
        remaining_X: The input data of the remaining instances.
        remaining_y: The target labels of the remaining instances.
    """
    X = np.array(X)
    y = np.array(y)

    unique_labels, first_occurrences, label_counts = np.unique(y, return_index=True, return_counts=True)
    if np.any(label_counts < n):
        raise ValueError("Value of n is bigger than the number of some labels")

    mask = np.zeros(len(y), dtype=bool)
    for label, start in zip(unique_labels, first_occurrences):
        label_positions = np.where(y == label)[0]
        selected_positions = label_positions[:n]
        mask[selected_positions] = True

    selected_X, selected_y = X[mask], y[mask]
    remaining_X, remaining_y = X[~mask], y[~mask]

    return selected_X, selected_y, remaining_X, remaining_y
