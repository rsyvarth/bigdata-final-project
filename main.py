import numpy as np
from sklearn import cross_validation, linear_model, preprocessing, svm

RUN_OUTPUT = False
RUN_OUTPUT = True
RUN_TEST_SPLIT = not RUN_OUTPUT

# Load training data
train_labels = np.loadtxt(open("data/train.csv"), delimiter=',', usecols=[0], skiprows=1)
train_data = np.loadtxt(open("data/train.csv"), delimiter=',', usecols=range(1, 9), skiprows=1)

# Load data we should predict
predict_data = np.loadtxt(open("data/test.csv"), delimiter=',', usecols=range(1, 9), skiprows=1)

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
    (linear_model.LogisticRegression(C=3.5), 1),
    (svm.SVC(kernel='rbf', C=2.2, probability=True), 0),
]

# Run the test split version
if RUN_TEST_SPLIT:
    # transform using our one-hot encoding
    train_data = encoder.transform(train_data)

    # split up the test set
    sdata_train, sdata_validate, slabel_train, slabel_validate = cross_validation.train_test_split(train_data, train_labels, test_size=.40, random_state=4)
    

    test_predictions = [0.0 for i in range(0, sdata_validate.shape[0])]
    for model in models:
        # fit to our training set
        model[0].fit(sdata_train, slabel_train)

        # predict our validation set
        preds = model[0].predict_proba(sdata_validate)
        for i, pred in enumerate(preds):
            test_predictions[i] = test_predictions[i] + pred[1]*model[1]

    # calculate error
    err = 0
    count = 0
    for i, pred in enumerate(test_predictions):
        err += abs(pred - slabel_validate[i])
        count = count + 1

    print "Accuracy, %s" % ((1 - (err / count)))

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

