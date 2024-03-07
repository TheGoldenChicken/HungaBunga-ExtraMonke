import sklearn.model_selection
import numpy as np
import traceback

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit as sss, ShuffleSplit as ss, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import model_selection

from pprint import pprint
from collections import Counter
from multiprocessing import cpu_count
from time import time
from tabulate import tabulate
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# nan = float('nan')

TREE_N_ENSEMBLE_MODELS = [RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, DecisionTreeRegressor,ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor]

class GridSearchCVProgressBar(sklearn.model_selection.GridSearchCV):
    def _get_param_iterator(self):
        iterator = super(GridSearchCVProgressBar, self)._get_param_iterator()
        iterator = list(iterator)

        # Sanity check cv
        cv = sklearn.model_selection._split.check_cv(self.cv, None)

        # Get max value for progress bar (how many models we must fit total)
        n_candidates = len(iterator)
        n_splits = getattr(cv, 'n_splits', 3)
        max_value = n_candidates * n_splits

        class ParallelProgressBar(sklearn.model_selection._search.Parallel):
            def __call__(self, iterable):
                iterable = tqdm(iterable, total=max_value)
                iterable.set_description("GridSearchCV")
                return super(ParallelProgressBar, self).__call__(iterable)

        # TODO: Find another way of doing this rather than overwriting sklearn inbuilt functions
        sklearn.model_selection._search.Parallel = ParallelProgressBar

        return iterator


class RandomizedSearchCVProgressBar(sklearn.model_selection.RandomizedSearchCV):
    def _get_param_iterator(self):
        iterator = super(RandomizedSearchCVProgressBar, self)._get_param_iterator()
        iterator = list(iterator)
        
        cv = sklearn.model_selection._split.check_cv(self.cv, None)

        n_candidates = len(iterator)
        n_splits = getattr(cv, 'n_splits', 3)
        max_value = n_candidates * n_splits

        class ParallelProgressBar(sklearn.model_selection._search.Parallel):
            def __call__(self, iterable):
                iterable = tqdm(iterable, total=max_value)
                iterable.set_description("RandomizedSearchCV")
                return super(ParallelProgressBar, self).__call__(iterable)

        sklearn.model_selection._search.Parallel = ParallelProgressBar

        return iterator


def upsample_indices_clf(inds, y):
    """
    Upsample to prevent majority classes
    """
    # Sanity check to ensure we have as many indices as data points
    assert len(inds) == len(y)

    # Get count of majority class
    countByClass = dict(Counter(y))
    maxCount = max(countByClass.values())

    extras = []

    # Iterate over classes and their counts
    for klass, count in countByClass.items():
        # If majority class, do nothing
        if maxCount == count:
            continue

        # Else: calculate ratio **rounded down**
        ratio = int(maxCount / count)
        cur_inds = inds[y == klass]
        # Add equal amount of extra samples from current class from each current data point
        # Then add maxCount % count extra samples from current class (chosen randomly from given x's) 
        extras.append(np.concatenate((np.repeat(cur_inds, ratio - 1), np.random.choice(cur_inds, maxCount - ratio * count, replace=False))))
    return np.concatenate([inds] + extras)


def cv_clf(x, y, test_size = 0.2, n_splits = 5, random_state=None, doesUpsample = True):
    """
    Return shuffled and stratified (class distributions same across folds) indices for train and test
    Possible to do upsampling (artifially make class distribution be uniform) if classes are not equally distributed by default
    """
    sss_obj = sss(n_splits, test_size=test_size, random_state=random_state).split(x, y)

    if not doesUpsample:
        yield sss_obj

    for train_inds, valid_inds in sss_obj:
        yield (upsample_indices_clf(train_inds, y[train_inds]), valid_inds)


def cv_reg(x, test_size = 0.2, n_splits = 5, random_state=None):
    """
    Return regular cross-validation object (indices) - meant for regression
    Used for regression, since we can't do upsampling (easily) there
    Nor does stratified cross-validation make sense
    """
    return ss(n_splits, test_size=test_size, random_state=random_state).split(x)


def timeit(klass, params, x, y):
    """
    Meant to time the final fit of a model, so not the search itself.
    """

    start = time()
    clf = klass(**params)
    clf.fit(x, y)
    return time() - start


def handle_NaN
    # kill methods:
    # row_kill - Removes rows that have NaN
    # col_kill - Removes cols that have NaN
    # smart_kill - Row or Col kill with whichever kills the least
    # extra: replace killed - keep track of rows that were removed so you can add them back using other rows

    # impute methods:
    # monke - replace with 0
    # mean - replace with mean of data
    # random - replace with random value of feature taken from other data point (at random)

    # Let be:
    # Returns the same (why would you use this then, lmao)

def main_loop(
    models_n_params, x, y, isClassification, test_size = 0.2, n_splits=5,
    random_state=None, upsample=True, scoring=None, verbose=True, n_jobs=cpu_count() - 1,
    brain=False, grid_search=True):

    def cv_():
        if isClassification:
            return cv_clf(x, y, test_size, n_splits, random_state, upsample)
        else:
            return cv_reg(x, test_size, n_splits, random_state)

    res = []
    exception_logs = []

    num_features = x.shape[1]

    # No custom scoting is defined
    if scoring is None:
        'accuracy' if isClassification else 'neg_mean_squared_error'
    else:
        scoring = scoring
    # scoring = scoring or ('accuracy' if isClassification else 'neg_mean_squared_error')

    
    if brain:
        print('Scoring criteria:', scoring)

    # Iterate through each class and parameter set of the models given tuple(sklearn_model, dict{params})
    for i, (clf_Klass, parameters) in enumerate(tqdm(models_n_params)):


        try:
            if brain:
                print('-'*15, 'model %d/%d' % (i+1, len(models_n_params)), '-'*15)
                print(clf_Klass.__name__)

            # TODO: Find way to move n_clusters to not the 'main loop', should rather have models 'configured' beforehand!
            if clf_Klass == KMeans:
                parameters['n_clusters'] = [len(np.unique(y))]

            elif clf_Klass in TREE_N_ENSEMBLE_MODELS:
                parameters['max_features'] = [v for v in parameters['max_features'] if v is None or isinstance(v, str) or v<=num_features]
            
            if grid_search:
                clf_search = GridSearchCVProgressBar(clf_Klass(), parameters, scoring, cv=cv_(), n_jobs=n_jobs)

            else:
                clf_search = RandomizedSearchCVProgressBar(clf_Klass(), parameters, scoring, cv=cv_(), n_jobs=n_jobs)

            # TODO: Consider if this isn't totally redundant
            clf_search.fit(x, y)
            timespent = timeit(clf_Klass, clf_search.best_params_, x, y)

            if brain:
                print('best score:', clf_search.best_score_, 'time/clf: %0.3f seconds' % timespent)
                print('best params:')
                pprint(clf_search.best_params_)

            if verbose:
                print('validation scores:', clf_search.cv_results_['mean_test_score'])
                print('training scores:', clf_search.cv_results_['mean_train_score'])
            
            # Append best estimator, best score and time spent getting it.
            res.append((clf_search.best_estimator_, clf_search.best_params_, clf_search.best_score_, timespent))

        except Exception as e:
            if verbose:
                traceback.print_exc()

            exception_logs.append(e)
            res.append((clf_Klass(), [],  -np.inf, np.inf))

    winner_ind = np.argmax([v[1] for v in res])
    winner = res[winner_ind][0]

    if brain:
        print('='*60)
        # Not gonna touch the formatting here, it works as it should
        print(tabulate([[m.__class__.__name__, '%.3f'%s, '%.3f'%t] for m, s, t in res], headers=['Model', scoring, 'Time/clf (s)']))
        print('='*60)
        print(f"The winner is: {winner.__class__.__name__} with score {res[winner_ind][1]:.3f.}")
    

    return winner, res


# Tests the cross-validation
if __name__ == '__main__':
    y = np.array([0,1,0,0,0,3,1,1,3])
    x = np.zeros(len(y))
    for t, v in cv_reg(x):
        print(v,t)
    for t, v in cv_clf(x, y, test_size=5):
        print(v,t)