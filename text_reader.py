

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np


if __name__ == '__main__':

    dataset = load_files("data/", shuffle=False)
    dataset_test = load_files("data_amazon/", shuffle=False)

    print ("sampels: %d" % len(dataset.data))


    # obj1 :parameter change
    tv = TfidfVectorizer()
    pipeline = Pipeline([
        ('vect', tv),
        ('clf', LinearSVC()),
    ])

    parameters = {
        'vect__ngram_range': [(1,1), (1, 2), (2,2)],
        'vect__min_df': ( 0.0067, 0.0065, 0.0063),
        'vect__max_df' : ( 0.945, 0.94, 0.935),
        'vect__norm': ('l1', 'l2'),
        'clf__C': (1,10),
        'clf__loss': ('hinge', 'squared_hinge'),
        'clf__tol': (1e-3, 1e-4)
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search.fit(dataset.data, dataset.target)


    feature_names = np.asarray(grid_search.best_estimator_.named_steps['vect'].get_feature_names())


    # obj2 : show features
    top10 = np.argsort(grid_search.best_estimator_.named_steps['clf'].coef_[0])[-10:]
    worst10 = np.argsort(grid_search.best_estimator_.named_steps['clf'].coef_[0])[:10]


    print [str(x) for x in feature_names[top10]]
    print [str(x) for x in feature_names[worst10]]


    #
    # for i, category in enumerate(dataset.target):
    #     print i
    #     print category
    #     print ("%s: %s" % (category, " ".join(feature_names[top10])))

    best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    print score


    y_predicted = grid_search.predict(dataset_test.data)

    # Print the classification report
    print(metrics.classification_report(dataset_test.target, y_predicted,
                                        target_names=dataset.target_names))


    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(dataset_test.target, y_predicted)
    print(cm)

    print "=========="
    # obj3: Show misclassified examples

    for i in range(0, len(dataset_test.data)):
        if dataset_test.target[i] != y_predicted[i]:
            print ("Original %s: %s" % (dataset_test.target[i], dataset_test.data[i]))
