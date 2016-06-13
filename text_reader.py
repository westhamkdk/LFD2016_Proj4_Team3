

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
import numpy as np


if __name__ == '__main__':

    dataset = load_files("data/", shuffle=False)
    dataset_test = load_files("data_amazon/", shuffle=False)

    print ("sampels: %d" % len(dataset.data))


    # obj1 :parameter change
    tv = TfidfVectorizer(min_df=3, max_df=0.95)
    pipeline = Pipeline([
        ('vect', tv),
        ('clf', LinearSVC(C=1.0)),
    ])

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (2,2)],
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search.fit(dataset.data, dataset.target)


    feature_names = np.asarray(grid_search.best_estimator_.named_steps['vect'].get_feature_names())


    # obj2 : show features
    top10 = np.argsort(grid_search.best_estimator_.named_steps['clf'].coef_[0])[-10:]
    worst10 = np.argsort(grid_search.best_estimator_.named_steps['clf'].coef_[0])[:10]

    print feature_names[top10]
    print feature_names[worst10]


    #
    # for i, category in enumerate(dataset.target):
    #     print i
    #     print category
    #     print ("%s: %s" % (category, " ".join(feature_names[top10])))


    y_predicted = grid_search.predict(dataset_test.data)

    # Print the classification report
    print(metrics.classification_report(dataset_test.target, y_predicted,
                                        target_names=dataset.target_names))


    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(dataset_test.target, y_predicted)
    print(cm)

    print "=========="
    # obj3: Show misclassified examples
