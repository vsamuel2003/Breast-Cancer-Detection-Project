from sklearn import linear_model, datasets, model_selection, metrics, tree, ensemble

cancer_data = datasets.load_breast_cancer()
COLUMN_NAMES = cancer_data.feature_names
X = cancer_data.data
y = cancer_data.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)


def randforest(X_train, X_test, y_train, y_test):
    forest = ensemble.RandomForestClassifier(n_estimators=10,
                                             criterion='entropy',
                                             bootstrap=False,
                                             random_state=0)
    forest.fit(X_train, y_train)
    acc = forest.score(X_test, y_test)
    predictions = forest.predict(X_test)
    matrix = metrics.confusion_matrix(y_test, predictions)
    print(f' Predictions: {predictions}')
    print(f' Accuracy: {acc}')
    print(f' Confusion Matrix: {matrix}')

print(randforest(X_train, X_test, y_train, y_test))