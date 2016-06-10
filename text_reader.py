

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


if __name__ == '__main__':

    dataset = load_files("data/", shuffle=False)
    dataset_test = load_files("data_amazon/", shuffle=False)

    print ("sampels: %d" % len(dataset.data))
    
    # obj1 : show features



    # obj2 :parameter change
    tv = TfidfVectorizer(min_df=3, max_df=0.95)
    pipeline = Pipeline([
        ('vect', tv),
        ('clf', LinearSVC(C=1.0)),
    ])

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        # 'vect__ngram_range': [(1, 2)],

    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search.fit(dataset.data, dataset.target)

    print(grid_search.grid_scores_)

    y_predicted_train = grid_search.predict(dataset.data)

    y_predicted = grid_search.predict(dataset_test.data)

    # Print the classification report
    print(metrics.classification_report(dataset_test.target, y_predicted,
                                        target_names=dataset.target_names))


    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(dataset_test.target, y_predicted)
    print(cm)

    print tv.get_feature_names()

    print "=========="
    # obj3: Show misclassified examples
