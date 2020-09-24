import numpy as np
from ml_pipeline import preprocessing, representation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def pipeline(preprocessor, representation, classifier):
    return Pipeline([('prep', preprocessor),
                     ('frm', representation),
                     ('clf', classifier)])


def combined_pipeline(prep1, repr1, prep2, repr2, classifier):
    combined_features = FeatureUnion([
        ('token_features', Pipeline([('prep1', prep1), ('repr1', repr1)])),
        ('polarity_features', Pipeline([('prep2', prep2), ('repr2', repr2)]))])
    return Pipeline([('features', combined_features),
                     ('clf', classifier)])


# ------------- parametrization ---------------------------


def svm_clf_grid_parameters():
    """Example parameters for svm.LinearSVC grid search

    The preprocessor and formatter can also be parametrized through the prefixes 'prep' and 'frm', respectively."""
    return {'clf__class_weight': (None, 'balanced'),
            'clf__dual': (True, False),
            'clf__C': (0.1, 1, 10)}


# ------------- standard pipelines ---------------------------------
def naive_bayes_counts():
    return pipeline(preprocessing.std_prep(),
                    representation.count_vectorizer({'min_df': 1}),
                    MultinomialNB())


def naive_bayes_tfidf():
    return pipeline(preprocessing.std_prep(),
                    representation.tfidf_vectorizer(),
                    MultinomialNB())


def naive_bayes_counts_bigram():
    return pipeline(preprocessing.std_prep(),
                    representation.count_vectorizer({'min_df': 1, 'ngram_range': (2, 2)}),
                    MultinomialNB())


def naive_bayes_counts_trigram():
    return pipeline(preprocessing.std_prep(),
                    representation.count_vectorizer({'min_df': 1, 'ngram_range': (3, 3)}),
                    MultinomialNB())


def naive_bayes_counts_lex():
    return pipeline(preprocessing.lex_prep(),
                    representation.count_vectorizer({'min_df': 1}),
                    MultinomialNB())


def svm_libsvc_counts():
    return pipeline(preprocessing.std_prep(),
                    representation.count_vectorizer(),
                    svm.LinearSVC(max_iter=10000,
                                  dual=False, C=0.1))


def svm_libsvc_counts_bigram():
    return pipeline(preprocessing.std_prep(),
                    representation.count_vectorizer({'min_df': 1, 'ngram_range': (2, 2)}),
                    svm.LinearSVC(max_iter=10000,
                                  dual=False,
                                  C=0.1))


def svm_libsvc_tfidf():
    return pipeline(preprocessing.std_prep(),
                    representation.tfidf_vectorizer(),
                    svm.LinearSVC(max_iter=10000,
                                  dual=False,
                                  C=0.1))


def svm_libsvc_embed():
    return pipeline(preprocessing.std_prep(),
                    representation.text2embeddings('wiki-news'),
                    svm.LinearSVC(max_iter=10000,
                                  dual=False,
                                  C=0.1))


def svm_sigmoid_embed():
    return pipeline(preprocessing.std_prep(),
                    representation.text2embeddings('glove'),
                    svm.SVC(kernel='sigmoid',
                            gamma='scale'))


def random_forest_embed():
    return pipeline(preprocessing.std_prep(),
                    representation.text2embeddings('glove'),
                    RandomForestClassifier(random_state=42, n_estimators=1000))


def random_forest_tfidf():
    return pipeline(preprocessing.std_prep(),
                    representation.tfidf_vectorizer(),
                    RandomForestClassifier(random_state=42, n_estimators=1000))


# -------------- STOPWORDS PIPELINES ----------------------


def naive_bayes_tfidf_stopwords():
    return pipeline(preprocessing.std_prep_stop(),
                    representation.tfidf_vectorizer(),
                    MultinomialNB())


def svm_libsvc_tfidf_stopwords():
    return pipeline(preprocessing.std_prep_stop(),
                    representation.tfidf_vectorizer(),
                    svm.LinearSVC(max_iter=10000,
                                  dual=False, C=0.1))


def random_forest_tfidf_stopwords():
    return pipeline(preprocessing.std_prep_stop(),
                    representation.tfidf_vectorizer(),
                    RandomForestClassifier(random_state=42, n_estimators=1000))
