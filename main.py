import dataHandling as dh
import model as model

def main():
    data = dh.load_images_from_directories()

    x_train, x_test, y_train, y_test = dh.split_data(data)

    model.svm_classifier(x_train, x_test, y_train, y_test)
                                               

if __name__ == '__main__':
    main()