import numpy as np
from sklearn import cross_validation, linear_model, preprocessing, svm, ensemble, naive_bayes, neighbors, feature_selection, metrics
import sys

RUN_OUTPUT = False
RUN_OUTPUT = True
RUN_TEST_SPLIT = not RUN_OUTPUT

COLS = [1,2,3,4,5,7,8,9]

def main():
    print "Loading train data"
    # Load training data
    train_labels = np.loadtxt(open("data/train.csv"), delimiter=',', usecols=[0], skiprows=1)
    train_data = np.loadtxt(open("data/train.csv"), delimiter=',', usecols=COLS, skiprows=1)
    train_data = feature_select_combine(train_data)

    print "Loading test data"
    # Load data we should predict
    # Note: we still use 1-10 here because row 0 is an ID in this dataset (not the result we should be predicting)
    predict_data = np.loadtxt(open("data/test.csv"), delimiter=',', usecols=COLS, skiprows=1)
    predict_data = feature_select_combine(predict_data)

    # Setup the models, each one is format (model, weight)
    # Notes on parameters:
    # C: inverse regularization strength - larger number -> more fit data (but not overfit hopefully!)
    # 
    models = [
        (linear_model.LogisticRegression(C=4.4), 0.85),
        (ensemble.RandomForestClassifier(), 0.15),
        # (svm.SVC(kernel='rbf', C=1.4, probability=True), 0.05),
    ]

    print "Running feature selection"
    train_data_scaled = train_data / train_data.max(axis=0)
    print train_data_scaled[0]

    selector = feature_selection.RFE(models[0][0], 30, step=1, verbose=1)
    out = selector.fit(train_data_scaled, train_labels)

    print out.support_
    print out.ranking_

    # Filter out unwanted featured
    train_data = train_data[:, out.support_]
    predict_data = predict_data[:, out.support_]

    # Make a super big combo dataset so we can make our onehot encoder learn all of our 
    total_data = []
    for dataset in [train_data, predict_data]:
        for row in dataset:
            total_data.append(row)

    print "SUPER HOT"

    # Setup one hot encoding
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(total_data)

    # Run the test split version
    if RUN_TEST_SPLIT:

        print "Running evaluation of new fancy model"
        # transform using our one-hot encoding
        train_data = encoder.transform(train_data)

        acc_array = []
        for rand in range(0, 10):
            # split up the test set
            sdata_train, sdata_validate, slabel_train, slabel_validate = cross_validation.train_test_split(train_data, train_labels, test_size=.4, random_state=rand*44)

            preds = test_models(models, sdata_train, slabel_train, sdata_validate)
            
            acc = calculate_accuracy(models, preds, sdata_validate, slabel_validate)
            acc_array.append(acc)

        print "FINAL ACCURACY AFTER 10x CV"
        print "Mean: %s" % (sum(acc_array)/len(acc_array))
        print "Median: %s" % np.median(np.array(acc_array))

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

def test_models(models, sdata_train, slabel_train, sdata_validate):
    
    preds = []
    for model_id, model in enumerate(models):
        # fit to our training set
        model[0].fit(sdata_train, slabel_train)

        # predict our validation set
        preds.append(model[0].predict_proba(sdata_validate))

    return preds

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

        print "Model %s Accuracy, %s %s" % (model_id, metrics.auc(slabel_validate, test_predictions_curr_model, reorder=True), (1 - (err / count)))

    # calculate error
    err = 0
    count = 0
    for i, pred in enumerate(test_predictions):
        err += abs(pred - slabel_validate[i])
        count = count + 1

    print "Accuracy, %s %s" % (metrics.auc(slabel_validate, test_predictions, reorder=True), (1 - (err / count)))
    return metrics.auc(slabel_validate, test_predictions, reorder=True)

def feature_select_combine(data):
    # combine features
    num_features = data.shape[1]
    data = data.tolist()

    for row in data:
        for i in range(0, num_features):
            for j in range(i, num_features):
                row.append(row[i]*10 + row[j])

    return np.array(data)

main()