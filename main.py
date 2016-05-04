import numpy as np
from sklearn import cross_validation, linear_model, preprocessing

RUN_OUTPUT = False
# RUN_OUTPUT = True
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

# Setup the main model
model = linear_model.LogisticRegression(C=2.5)

# Run the test split version
if RUN_TEST_SPLIT:
    # transform using our one-hot encoding
    train_data = encoder.transform(train_data)

    # split up the test set
    sdata_train, sdata_validate, slabel_train, slabel_validate = cross_validation.train_test_split(train_data, train_labels, test_size=.20, random_state=4)
    
    # firt to our training set
    model.fit(sdata_train, slabel_train) 
    
    # predict our validation set
    preds = model.predict_proba(sdata_validate)

    # calculate error
    err = 0
    count = 0
    for i, pred in enumerate(preds):
        err += abs(pred[1] - train_labels[i])
        count = count + 1
    print "Accuracy: %s" % (1 - (err / count))

# Run the actual prediction for submission
if RUN_OUTPUT:
    # transform using our one-hot encoding
    train_data = encoder.transform(train_data)
    predict_data = encoder.transform(predict_data)
    
    # fit the whole dataset
    model.fit(train_data, train_labels)

    # predict things!
    test_predictions = model.predict_proba(predict_data)

    # output our predictions
    with open("output.csv", 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(test_predictions):
            f.write("%s,%s\n" % (i + 1, pred[1]))

