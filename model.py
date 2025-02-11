from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

def svm_classifier(x_train, x_test, y_train, y_test):
    svc = svm.SVC(probability=True)
    svc.fit(X=x_train, y=y_train)
    print("Trained")

    y_pred = svc.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=['AI', 'Real']))

    accuracy = accuracy_score(y_pred, y_test)

    print(f"The model is {accuracy * 100}% accurate")
