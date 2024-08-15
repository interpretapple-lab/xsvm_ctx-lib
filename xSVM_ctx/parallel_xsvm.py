import os
import sys
import numpy as np
from joblib import Parallel, delayed
import warnings

_path_to_lib_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _path_to_lib_ not in sys.path:
    sys.path.insert(0, _path_to_lib_)
else:
    _path_to_lib_ = None

from xsvmlib.xsvmc import xSVMC

if _path_to_lib_:
    sys.path.remove(_path_to_lib_)
del _path_to_lib_

class contextualized_xSVMC(xSVMC):
    """ Contextualized Explainable Support Vector Machine Classification

    This class is the implementation of the XSVM@ctx model proposed in [1], for the contextualization of *support vector
    machine* (SVM)[2] models. The model utilizes multiple SVMs, with each SVM corresponding to a specific context,
    allowing for a parallelized approach to SVM classification.

    This implementation utilizes the *explainable SVM classification* (xSVMC) model proposed in [2], which itself
    expands upon the SVM classification process described in [3]. It uses the most influential vectors (MISVs) for
    contextualizing the evaluations in order to provide contrasting explanations [3].

    Similar to the xSVMC model, this implementation is based on Scikit-learn SVC class.

    References:
        [1] M. Loor, A. Tapia-Rosero and G. De Tré. Contextual Boosting to Explainable SVM Classification.
            Massanet, S., Montes, S., Ruiz-Aguilera, D., González-Hidalgo, M. (eds) Fuzzy Logic and Technology, and
            Aggregation Operators. EUSFLAT AGOP 2023 2023. Lecture Notes in Computer Science, vol 14069. Springer, Cham.
            https://doi.org/10.1007/978-3-031-39965-7_40

        [2] V.N.Vapnik,The Nature of Statistical Learning Theory. Springer-Verlag, New York, NY, USA, 1995.
            https://dx.doi.org/10.1007/978-1-4757-3264-1

        [3] M. Loor and G. De Tré. Contextualizing Support Vector Machine Predictions.
            International Journal of Computational Intelligence Systems, Volume 13, Issue 1, 2020,
            Pages 1483-1497,  ISSN 1875-6883, https://doi.org/10.2991/ijcis.d.200910.002
    """
    def __init__(
            self,
            *,
            C=1.0,
            kernel="rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape="ovo",
            break_ties=False,
            random_state=None,
            k=1
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

        if not isinstance(k, int):
            raise ValueError("K parameter must be an integer")
        elif k < 1:
            raise ValueError("K parameter cannot lower than 0")
        self.k = k

    def __get_attribute(self, attr_name):
        """ Gets the attribute from all contextualized classifiers.

        Parameters:
            attr_name: the attribute to be retrieved.

        Returns:
            attr_dict: Dictionary with the attribute values for each context.
        """
        if self.fit_status_ != 0:
            raise AttributeError(
                f"This parallel_xSVC instance is not fitted yet. Call 'fit' with appropriate arguments before accessing '{attr_name}'.")
        return {context: getattr(clf, attr_name) for context, clf in self.contextualized_clfs_.items()}

    @property
    def support_(self):
        return self.__get_attribute('support_')

    @property
    def support_vectors_(self):
        return self.__get_attribute('support_vectors_')

    @property
    def n_support_(self):
        return self.__get_attribute('n_support_')

    @property
    def dual_coef_(self):
        return self.__get_attribute('dual_coef_')

    @property
    def intercept_(self):
        return self.__get_attribute('intercept_')

    @property
    def class_weight_(self):
        return self.__get_attribute('class_weight_')

    @property
    def n_features_in_(self):
        return self.__get_attribute('n_features_in_')

    @property
    def feature_names_in_(self):
        return self.__get_attribute('feature_names_in_')

    @property
    def n_iter_(self):
        return self.__get_attribute('n_iter_')

    @property
    def shape_fit_(self):
        return self.__get_attribute('shape_fit_')

    @property
    def coef_(self):
        return self.__get_attribute('coef_')

    @property
    def probA_(self):
        return self.__get_attribute('probA_')

    @property
    def probB_(self):
        return self.__get_attribute('probB_')

    @property
    def classes_(self):
        return self.__get_attribute('classes_')

    @property
    def contexts_(self):
        contexts = [context for context in self.contextualized_clfs_.keys()]
        return np.array(contexts)

    def __fit(self, context, X, y):
        """ Fits a classifier in a given context.

        Parameters:
            context: The context in which the classifier will be fitted.
            X: The input data of shape (n_samples, n_features).
            y: The target classes of shape (n_samples,).

        Returns:
            context: The context in which the classifier was fitted.
            clf: The xSVMC model fitted in the given context.
        """
        unique_classes = np.unique(y)
        if len(unique_classes) <= 1:
            return context, None

        clf = xSVMC(**self.get_params())
        clf.fit(X, y)
        return context, clf

    # fit
    def fit(self, contextualized_X, contextualized_y, n_jobs=-1, verbose=0, backend='threading'):
        """ Trains the classifier in each context.

        Parameters:
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).
            contextualized_y: A dictionary with the shape {context: y} where y is the target classes of shape
            (n_samples,).
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            The fitted instance of the contextualized model
        """
        self.contextualized_clfs_ = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(self.__fit)(context, contextualized_X[context], contextualized_y[context]) for context in
            contextualized_X.keys()
        )
        self.contextualized_clfs_ = {context: clf for context, clf in self.contextualized_clfs_ if clf is not None}
        self.fit_status_ = 0
        return self

    # predict
    def __predict(self, context, clf, contextualized_X):
        """ Performs the prediction of the classes for the input data in a given context.

        Parameters:
            context: The context in which the classifier will predict.
            clf: The classifier to be used for prediction.
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).

        Returns:
            context: The context in which the classifier predicted.
            predictions: ndarray of shape (n_samples) consisting of the classes predicted for X in the specified context.
        """
        if context not in contextualized_X:
            return context, None
        X = contextualized_X[context]
        return context, clf.predict(X)

    def predict(self, contextualized_X, n_jobs=-1, verbose=0, backend='threading'):
        """ Performs a contextualized prediction of the contextualized input data.

        Parameters:
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            predictions: A dictionary with the shape {context: predictions} where predictions is a ndarray of shape
            (n_samples) consisting of the classes predicted for X in the specified context.
        """
        predictions = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(self.__predict)(context, clf, contextualized_X) for context, clf in
            self.contextualized_clfs_.items()
        )
        return {context: pred for context, pred in predictions if pred is not None}

    # predict_with_context
    def __predict_with_context(self, context, clf, contextualized_X):
        """ Performs an augmented prediction of the top-K classes for the input data in a given context.

        Parameters:
            context: The context in which the classifier will predict.
            clf: The classifier to be used for prediction.
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).

        Returns:
            context: The context in which the classifier predicted.
            predictions: ndarray of shape (n_samples, k) consisting of the top-K classes predicted for X in the
            specified context.
        """
        if context not in contextualized_X:
            return context, None
        X = contextualized_X[context]
        return context, clf.predict_with_context(X)

    def predict_with_context(self, contextualized_X, n_jobs=-1, verbose=0, backend='threading'):
        """ Performs an augmented contextualized prediction of the top-K classes for each sample in the contextualized
        input data.

        Parameters:
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            predictions: A dictionary with the shape {context: predictions} where predictions is a ndarray of shape
            (n_samples, k) consisting of the top-K classes predicted for X in the specified context.
        """
        predictions = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(self.__predict_with_context)(context, clf, contextualized_X) for context, clf in
            self.contextualized_clfs_.items()
        )
        return {context: pred for context, pred in predictions if pred is not None}

    # predict_with_context_by_voting
    def __predict_with_context_by_voting(self, context, clf, contextualized_X):
        """ Performs an augmented prediction of the top-K classes for the input data in a given context.

        Parameters:
            context: The context in which the classifier will predict.
            clf: The classifier to be used for prediction.
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).

        Returns:
            context: The context in which the classifier predicted.
            predictions: ndarray of shape (n_samples, k) consisting of the top-K classes predicted for X in the
            specified context.
        """
        if context not in contextualized_X:
            return context, None
        X = contextualized_X[context]
        return context, clf.predict_with_context_by_voting(X)

    def predict_with_context_by_voting(self, contextualized_X, n_jobs=-1, verbose=0, backend='threading'):
        """ Performs an augmented contextualized prediction of the top-K classes for each sample in the contextualized
        input data.

        Parameters:
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            predictions: A dictionary with the shape {context: predictions} where predictions is a ndarray of shape
            (n_samples, k) consisting of the top-K classes predicted for X in the specified context.
        """
        predictions = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(self.__predict_with_context_by_voting)(context, clf, contextualized_X) for context, clf in
            self.contextualized_clfs_.items()
        )
        return {context: pred for context, pred in predictions if pred is not None}

    # evaluate_all_memberships
    def __evaluate_all_memberships(self, context, clf, contextualized_X):
        """ Performs augmented evaluation of the proposition 'X IS A' for each class A learned during the training process
        for each sample in the input data in a given context.

        Parameters:
            context: The context in which the classifier will predict.
            clf: The classifier to be used for prediction.
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).

        Returns:
            context: The context in which the classifier predicted.
            evaluations: ndarray of shape (n_samples, n_classes) consisting of the augmented evaluations.
        """
        if context not in contextualized_X:
            return context, None
        X = contextualized_X[context]
        return context, clf.evaluate_all_memberships(X)

    def evaluate_all_memberships(self, contextualized_X, n_jobs=-1, verbose=0, backend='threading'):
        """ Performs augmented evaluation of the proposition 'X IS A' for each class A learned during the training process
        for each sample in the contextualized input data.

        Parameters:
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            evaluations: A dictionary with the shape {context: evaluations} where evaluations is a ndarray of shape
            (n_samples, n_classes) consisting of the augmented evaluations.
        """
        evaluations = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(self.__evaluate_all_memberships)(context, clf, contextualized_X) for context, clf in
            self.contextualized_clfs_.items()
        )
        return {context: evals for context, evals in evaluations if evals is not None}

    # decision_function
    def __decision_function(self, context, clf, contextualized_X):
        """ Evaluates the decision function for each sample in the input data in a given context.

        Parameters:
            context: The context in which the classifier will predict.
            clf: The classifier to be used for prediction.
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).

        Returns:
            context: The context in which the classifier predicted.
            decisions: ndarray consisting of the decision function of the sample for each class in the model.

        Notes:
            If decision_function_shape='ovr', the shape of decisions is (n_samples, n_classes).
            If decision_function_shape='ovo', the shape of decisions is (n_samples, n_classes * (n_classes - 1) / 2).
        """
        if context not in contextualized_X:
            return context, None
        X = contextualized_X[context]
        return context, clf.decision_function(X)

    def decision_function(self, contextualized_X, n_jobs=-1, verbose=0, backend='threading'):
        """ Evaluates the decision function for each sample in the contextualized input data.

        Parameters:
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            decisions: A dictionary with the shape {context: decisions} where decisions is a ndarray consisting of the
            decision function of the sample for each class in the model.

        Notes:
            If decision_function_shape='ovr', the shape of decisions is (n_samples, n_classes).
            If decision_function_shape='ovo', the shape of decisions is (n_samples, n_classes * (n_classes - 1) / 2).
        """
        decisions = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(self.__decision_function)(context, clf, contextualized_X) for context, clf in
            self.contextualized_clfs_.items()
        )
        return {context: dec for context, dec in decisions if dec is not None}

    # decision_function_with_context
    def __decision_function_with_context(self, context, clf, contextualized_X):
        """ Evaluates the decision function for each sample in the input data in a given context.

        Parameters:
            context: The context in which the classifier will predict.
            clf: The classifier to be used for prediction.
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).

        Returns:
            context: The context in which the classifier predicted.
            decisions: A 4-tuple (memberships, nonmemberships, pro_MISVs, con_MISVs) consisting of 4 ndarrays of shape
            (n_samples, n_classes * (n_classes-1) / 2).

        Notes:
            `decision_function_shape` is ignored for binary classification.
            decision_function_shape='ovo' is always used as multi-class strategy so the shape of decisions always is
            (n_samples, n_classes * (n_classes - 1) / 2).
        """
        if context not in contextualized_X:
            return context, None
        X = contextualized_X[context]
        return context, clf.decision_function_with_context(X)

    def decision_function_with_context(self, contextualized_X, n_jobs=-1, verbose=0, backend='threading'):
        """ Evaluates the decision function for each sample in the contextualized input data.

        Parameters:
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            decisions: A dictionary with the shape {context: decisions} where decisions is a 4-tuple  (memberships,
            nonmemberships, pro_MISVs, con_MISVs) consisting of 4 ndarrays of shape
            (n_samples, n_classes * (n_classes-1) / 2).

        Notes:
            `decision_function_shape` is ignored for binary classification.
            decision_function_shape='ovo' is always used as multi-class strategy so the shape of decisions always is
            (n_samples, n_classes * (n_classes - 1) / 2).
        """
        decisions = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(self.__decision_function_with_context)(context, clf, contextualized_X) for context, clf in
            self.contextualized_clfs_.items()
        )
        return {context: dec for context, dec in decisions if dec is not None}

    # predict_proba
    def __predict_proba(self, context, clf, contextualized_X):
        """ Compute probabilities of possible outcomes for each sample in the input data in a given context.

        Parameters:
            context: The context in which the classifier will predict.
            clf: The classifier to be used for prediction.
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).

        Returns:
            context: The context in which the classifier predicted.
            probabilities: ndarray of shape (n_samples, n_classes) consisting of the probabilities of each class in the
            model for each sample in the input data in the specified context.

        Notes:
            The attribute `probability` needs to be set to True during fit in order for the model to have the
            probability information computed.
        """
        if context not in contextualized_X:
            return context, None
        X = contextualized_X[context]
        return context, clf.predict_proba(X)

    def predict_proba(self, contextualized_X, n_jobs=-1, verbose=0, backend='threading'):
        """ Compute probabilities of possible outcomes for each sample in the contextualized input data.

        Parameters:
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            probabilities: A dictionary with the shape {context: probabilities} where probabilities is a ndarray of shape
            (n_samples, n_classes) consisting of the probabilities of each class in the model for each sample in the
            input data in that context.

        Notes:
            The attribute `probability` needs to be set to True during fit in order for the model to have the
            probability information computed.
        """
        probabilities = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(self.__predict_proba)(context, clf, contextualized_X) for context, clf in
            self.contextualized_clfs_.items()
        )
        return {context: prob for context, prob in probabilities if prob is not None}

    # predict_log_proba
    def __predict_log_proba(self, context, clf, contextualized_X):
        """ Compute log probabilities of possible outcomes for each sample in the input data in a given context.

        Parameters:
            context: The context in which the classifier will predict.
            clf: The classifier to be used for prediction.
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).

        Returns:
            context: The context in which the classifier predicted.
            probabilities: ndarray of shape (n_samples, n_classes) consisting of the log probabilities of each class in
            the model for each sample in the input data in the specified context.

        Notes:
            The attribute `probability` needs to be set to True during fit in order for the model to have the
            probability information computed.
        """
        if context not in contextualized_X:
            return context, None
        X = contextualized_X[context]
        return context, clf.predict_log_proba(X)

    def predict_log_proba(self, contextualized_X, n_jobs=-1, verbose=0, backend='threading'):
        """ Compute log probabilities of possible outcomes for each sample in the contextualized input data.

        Parameters:
            contextualized_X: A dictionary with the shape {context: X} where X is the input data of shape
            (n_samples, n_features).
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            probabilities: A dictionary with the shape {context: probabilities} where probabilities is a ndarray of shape
            (n_samples, n_classes) consisting of the log probabilities of each class in the model for each sample in the
            input data in that context.

        Notes:
            The attribute `probability` needs to be set to True during fit in order for the model to have the
            probability information computed.
        """
        log_probabilities = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(self.__predict_log_proba)(context, clf, contextualized_X) for context, clf in
            self.contextualized_clfs_.items()
        )
        return {context: log_prob for context, log_prob in log_probabilities if log_prob is not None}

    # score
    def __score(self, context, clf, X, y):
        """ Returns the mean accuracy on the given test data and labels in a given context.

        Parameters:
            context: The context in which the classifier will predict.
            clf: The classifier to be used for prediction.
            X: ndarray of shape (n_samples, n_features) consisting of the input data.
            y: ndarray of shape (n_samples,) consisting of the target classes.

        Returns:
            context: The context in which the classifier predicted.
            score: The mean accuracy of the classifier on the given test data and labels in the specified context.
        """
        return context, clf.score(X, y)

    def score(self, X, y, n_jobs=-1, verbose=0, backend='threading'):
        """ Returns the mean accuracy on the given test data and labels in each context.

        Parameters:
            X: ndarray of shape (n_samples, n_features) consisting of the input data.
            y: ndarray of shape (n_samples,) consisting of the target classes.
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            scores: A dictionary with the shape {context: score} where score is the mean accuracy of the classifier on
            the given test data and labels in that context.
        """
        scores = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(self.__score)(context, clf, X, y) for context, clf in
            self.contextualized_clfs_.items()
        )
        return {context: score for context, score in scores if score is not None}

    # score_with_context
    def __score_with_context(self, context, clf, X, y):
        """ Returns the mean accuracy on the given test data and labels in a given context.

        Parameters:
            context: The context in which the classifier will predict.
            clf: The classifier to be used for prediction.
            X: ndarray of shape (n_samples, n_features) consisting of the input data.
            y: ndarray of shape (n_samples,) consisting of the target classes.

        Returns:
            context: The context in which the classifier predicted.
            score: The mean accuracy of the classifier on the given test data and labels in the specified context.
        """
        predictions = clf.predict_with_context(X)
        score = sum(1 for x in range(len(y)) if y[x] == predictions[x][0].class_name) / len(y)
        return context, score

    def score_with_context(self, X, y, n_jobs=-1, verbose=0, backend='threading'):
        """ Returns the mean accuracy on the given test data and labels in each context.

        Parameters:
            X: ndarray of shape (n_samples, n_features) consisting of the input data.
            y: ndarray of shape (n_samples,) consisting of the target classes.
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            scores: A dictionary with the shape {context: score} where score is the mean accuracy of the classifier on
            the given test data and labels in that context.
        """
        scores = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(self.__score_with_context)(context, clf, X, y) for context, clf in
            self.contextualized_clfs_.items()
        )
        return {context: score for context, score in scores if score is not None}

    # score_with_context_by_voting
    def __score_with_context_by_voting(self, context, clf, X, y):
        """ Returns the mean accuracy on the given test data and labels in a given context.

        Parameters:
            context: The context in which the classifier will predict.
            clf: The classifier to be used for prediction.
            X: ndarray of shape (n_samples, n_features) consisting of the input data.
            y: ndarray of shape (n_samples,) consisting of the target classes.

        Returns:
            context: The context in which the classifier predicted.
            score: The mean accuracy of the classifier on the given test data and labels in the specified context.
        """
        predictions = clf.predict_with_context_by_voting(X)
        score = sum(1 for x in range(len(y)) if y[x] == predictions[x][0].class_name) / len(y)
        return context, score

    def score_with_context_by_voting(self, X, y, n_jobs=-1, verbose=0, backend='threading'):
        """ Returns the mean accuracy on the given test data and labels in each context.

        Parameters:
            X: ndarray of shape (n_samples, n_features) consisting of the input data.
            y: ndarray of shape (n_samples,) consisting of the target classes.
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            scores: A dictionary with the shape {context: score} where score is the mean accuracy of the classifier on
            the given test data and labels in that context.
        """
        scores = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(self.__score_with_context_by_voting)(context, clf, X, y) for context, clf in
            self.contextualized_clfs_.items()
        )
        return {context: score for context, score in scores if score is not None}

    # automatic_context_selection
    def automatic_context_selection(self, X, y, evaluation='context', n_jobs=-1, verbose=0, backend='threading'):
        """ Performs an automatic context selection based on the highest score.

        Parameters:
            X: ndarray of shape (n_samples, n_features) consisting of the input data.
            y: ndarray of shape (n_samples,) consisting of the target classes.
            evaluation: The evaluation function to be used for scoring the classifier.
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            context: The context with the highest score.

        Notes:
            If more than one context has the same score it returns the first context with the highest score and a
            warning is raised.
            This method for automatic context selection was proposed in [1].

        References:
            [1] M. Loor and G. De Tré. Automatic Context Selection in Explainable Support Vector Machine
                Classification. 2024 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), Yokohama, Japan, 2024,
                https://hdl.handle.net/1854/LU-01J3ZFMB2S6926B5TC2HTWGAXF
        """
        if evaluation == 'context':
            scores = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
                delayed(self.__score_with_context)(context, clf, X, y) for context, clf in
                self.contextualized_clfs_.items()
            )
        elif evaluation == 'context_by_voting':
            scores = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
                delayed(self.__score_with_context_by_voting)(context, clf, X, y) for context, clf in
                self.contextualized_clfs_.items()
            )
        else:
            scores = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
                delayed(self.__score)(context, clf, X, y) for context, clf in
                self.contextualized_clfs_.items()
            )

        contextualized_scores = {context: score for context, score in scores}
        max_value = max(contextualized_scores.values())
        max_keys = [key for key, value in contextualized_scores.items() if value == max_value]
        if len(max_keys) > 1:
            warnings.warn(
                "More than one context has the same score. It is recommended to increase the number of labeled objects.",
                Warning)

        return max_keys[0]

    def predict_with_unknown_context(self, X, labeled_X, y, n_jobs=-1, verbose=0, backend='threading'):
        """ Performs an automatic context selection and makes a prediction of the data in the selected context.

        Parameters:
            X: ndarray of shape (n_samples, n_features) consisting of the input data.
            labeled_X: ndarray of shape (n_samples, n_features) consisting of the labeled input data.
            y: ndarray of shape (n_samples,) consisting of the target classes.
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            predictions: ndarray of shape (n_samples, k) consisting of the top-K classes predicted for X in the
            detected context.
        """
        context = self.automatic_context_selection(labeled_X, y, evaluation="context", n_jobs=n_jobs, verbose=verbose,
                                                   backend=backend)
        print(f"Selected context: {context}")
        return self.contextualized_clfs_[context].predict_with_context(X)

    def predict_with_unknown_context_by_voting(self, X, labeled_X, y, n_jobs=-1, verbose=0, backend='threading'):
        """ Performs an automatic context selection and makes a prediction of the data in the selected context.

        Parameters:
            X: ndarray of shape (n_samples, n_features) consisting of the input data.
            labeled_X: ndarray of shape (n_samples, n_features) consisting of the labeled input data.
            y: ndarray of shape (n_samples,) consisting of the target classes.
            n_jobs: The number of jobs to run in parallel. -1 means using all processors.
            verbose: The verbosity level.
            backend: The parallelization backend to use.

        Returns:
            predictions: ndarray of shape (n_samples, k) consisting of the top-K classes predicted for X in the
            detected context.
        """
        context = self.automatic_context_selection(labeled_X, y, evaluation="context_by_voting", n_jobs=n_jobs,
                                                   verbose=verbose, backend=backend)

        print(f"Selected context: {context}")
        return self.contextualized_clfs_[context].predict_with_context_by_voting(X)
