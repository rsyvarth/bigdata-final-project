import numpy as np
from sklearn import cross_validation, linear_model, preprocessing, svm, ensemble, naive_bayes, neighbors

RUN_OUTPUT = False
RUN_OUTPUT = True
RUN_TEST_SPLIT = not RUN_OUTPUT

COLS = [1,2,3,4,5,7,8,9]

def main():
    # Load training data
    train_labels = np.loadtxt(open("data/train.csv"), delimiter=',', usecols=[0], skiprows=1)
    train_data = np.loadtxt(open("data/train.csv"), delimiter=',', usecols=COLS, skiprows=1)

    # Load data we should predict
    # Note: we still use 1-10 here because row 0 is an ID in this dataset (not the result we should be predicting)
    predict_data = np.loadtxt(open("data/test.csv"), delimiter=',', usecols=COLS, skiprows=1)

    # Make a super big combo dataset so we can make our onehot encoder learn all of our 
    total_data = []
    for dataset in [train_data, predict_data]:
        for row in dataset:
            total_data.append(row)

    # Setup one hot encoding
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(total_data)

    # Setup the models, each one is format (model, weight)
    # Notes on parameters:
    # C: inverse regularization strength - larger number -> more fit data (but not overfit hopefully!)
    # 
    models = [
        (linear_model.LogisticRegression(C=4.4), 0.8),
        (ensemble.RandomForestClassifier(), 0.1),
        (svm.SVC(kernel='rbf', C=1.4, probability=True), 0.1),

        # (ensemble.GradientBoostingClassifier(), 0.1),
        # (ensemble.AdaBoostClassifier(), 0),

        # (naive_bayes.BernoulliNB(), 1),
    ]

    # Run the test split version
    if RUN_TEST_SPLIT:
        # transform using our one-hot encoding
        train_data = encoder.transform(train_data)

        preds, sdata_validate, slabel_validate = test_models(models, train_data, train_labels)
        
        acc = calculate_accuracy(models, preds, sdata_validate, slabel_validate)

    # Run the actual prediction for submission
    if RUN_OUTPUT:
        # transform using our one-hot encoding
        train_data = encoder.transform(train_data)
        predict_data = encoder.transform(predict_data)
        
        # fit the whole dataset
        test_predictions = [0.0 for i in range(0, predict_data.shape[0])]
        for model in models:
            model[0].fit(train_data, train_labels)

            # predict things!
            preds = model[0].predict_proba(predict_data)
            for i, pred in enumerate(preds):
                test_predictions[i] = test_predictions[i] + pred[1]*model[1]

        # output our predictions
        with open("output.csv", 'w') as f:
            f.write("id,ACTION\n")
            for i, pred in enumerate(test_predictions):
                f.write("%s,%s\n" % (i + 1, pred))

def test_models(models, train_data, train_labels):
    # split up the test set
    sdata_train, sdata_validate, slabel_train, slabel_validate = cross_validation.train_test_split(train_data, train_labels, test_size=.4, random_state=44)

    preds = []
    for model_id, model in enumerate(models):
        # fit to our training set
        model[0].fit(sdata_train, slabel_train)

        # predict our validation set
        preds.append(model[0].predict_proba(sdata_validate))

    return preds, sdata_validate, slabel_validate

def calculate_accuracy(models, preds, sdata_validate, slabel_validate):
    test_predictions = [0.0 for i in range(0, sdata_validate.shape[0])]
    for model_id, model in enumerate(models):
        test_predictions_curr_model = [0.0 for i in range(0, sdata_validate.shape[0])]

        for i, pred in enumerate(preds[model_id]):
            test_predictions[i] = test_predictions[i] + pred[1]*model[1]
            test_predictions_curr_model[i] = pred[1]

        err = 0
        count = 0
        for i, pred in enumerate(test_predictions_curr_model):
            err += abs(pred - slabel_validate[i])
            count = count + 1

        print "Model %s Accuracy, %s" % (model_id, (1 - (err / count)))

    # calculate error
    err = 0
    count = 0
    for i, pred in enumerate(test_predictions):
        err += abs(pred - slabel_validate[i])
        count = count + 1

    print "Accuracy, %s" % ((1 - (err / count)))
    return (1 - (err / count))

main()