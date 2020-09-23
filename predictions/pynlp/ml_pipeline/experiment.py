import logging
import sys

from ml_pipeline import pipelines
from ml_pipeline import utils, cnn, preprocessing, pipeline_with_lexicon
from ml_pipeline.cnn import CNN
from tasks import vua_format as vf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
# handler = logging.FileHandler('experiment.log')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def run(task_name, data_dir, pipeline_name, print_predictions, error_analysis, remove_stopwords):
    logger.info('>> Running {} experiment'.format(task_name))
    tsk = task(task_name)
    logger.info('>> Loading data...')
    tsk.load(data_dir)
    logger.info('>> retrieving train/data instances...')
    train_X, train_y, test_X, test_y = utils.get_instances(tsk, split_train_dev=False)
    test_X_ref = test_X

    if remove_stopwords:
        if pipeline_name.startswith('cnn'):
            pipeline_name = pipeline_name.split('_')[0]
        pipeline_name = pipeline_name + '_stopwords'

    if pipeline_name.startswith('cnn'):
        pipe = cnn(pipeline_name)
        train_X, train_y, test_X, test_y = pipe.encode(train_X, train_y, test_X, test_y)
        logger.info('>> testing CNN...')
    else:
        pipe = pipeline(pipeline_name)

    logger.info('>> training pipeline ' + pipeline_name)
    pipe.fit(train_X, train_y)
    if pipeline_name == 'naive_bayes_counts_lex':
        logger.info("   -- Found {} tokens in lexicon".format(pipe.tokens_from_lexicon))

    logger.info('>> testing...')
    sys_y = pipe.predict(test_X)
    # logger.info(utils.print_prediction(test_X, test_y, sys_y))

    if print_predictions:
        logger.info('>> predictions')
        utils.print_all_predictions(test_X_ref, test_y, sys_y, logger)

    if error_analysis:
        # Used for error evaluation
        logger.info(utils.print_error_analysis(test_X, test_y, sys_y))
        # logger.info(utils.print_confusion_matrix(test_y, sys_y)) # Prints the confusion matrix

    utils.eval(test_y, sys_y, pipeline_name, data_dir)
    if pipeline_name.startswith('naive_bayes'):
        utils.important_features_per_class(pipe.named_steps.frm, pipe.named_steps.clf, n=10)


def task(name):
    if name == 'offenseval':
        return of.Offenseval()
    elif name == 'vua_format':
        return vf.VuaFormat()
    else:
        raise ValueError("task name is unknown. You can add a custom task in 'tasks'")


def cnn(name):
    if name == 'cnn_raw':
        return CNN()
    elif name == 'cnn_prep':
        return CNN(preprocessing.std_prep())
    elif name == 'cnn_stopwords':
        return CNN(preprocessing.std_prep_stop())
    else:
        raise ValueError("pipeline name is unknown.")


def pipeline(name):
    if name == 'naive_bayes_counts':
        return pipelines.naive_bayes_counts()
    elif name == 'naive_bayes_tfidf':
        return pipelines.naive_bayes_tfidf()
    elif name == 'naive_bayes_tfidf_stopwords':
        return pipelines.naive_bayes_tfidf_stopwords()
    elif name == 'naive_bayes_bigram':
        return pipelines.naive_bayes_bigram()
    elif name == 'naive_bayes_trigram':
        return pipelines.naive_bayes_trigram()
    elif name == 'naive_bayes_counts_lex':
        return pipeline_with_lexicon.naive_bayes_counts_lex()
    elif name == 'svm_libsvc_counts':
        return pipelines.svm_libsvc_counts()
    elif name == 'svm_libsvc_tfidf':
        return pipelines.svm_libsvc_tfidf()
    elif name == 'svm_libsvc_tfidf_stopwords':
        return pipelines.svm_libsvc_tfidf_stopwords()
    elif name == 'svm_libsvc_embed':
        return pipelines.svm_libsvc_embed()
    elif name == 'svm_sigmoid_embed':
        return pipelines.svm_sigmoid_embed()
    elif name == 'random_forest_tfidf':
        return pipelines.random_forest_tfidf()
    elif name == 'random_forest_tfidf_stopwords':
        return pipelines.random_forest_tfidf_stopwords()
    elif name == 'random_forest_embed':
        return pipelines.random_forest_embed()
    else:
        raise ValueError("pipeline name is unknown. You can add a custom pipeline in 'pipelines'")
