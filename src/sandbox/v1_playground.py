from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)


from ml_lib.cnn_model import CNN
from ml_lib.moleimages import MoleImages
from ml_lib.roc import plot_roc

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import os.path


train_data_dir = 'data_scaled/'
validation_data_dir = 'data_scaled_validation/'
nb_train_samples = 1853
nb_validation_samples = 204
batch_size = 32

mimg = MoleImages()

X_test, y_test = mimg.load_test_images('data_scaled_validation/benign', 'data_scaled_validation/malign')

X_train, y_train = mimg.load_test_images('data_scaled/benign', 'data_scaled/malign')

mycnn = CNN()

if os.path.isfile("models/model_1_test.h5"):
    mycnn.load_model("models/model_1_test.h5")

areas = []
accuracies = []

for i in range(0, 30):
    score = mycnn.fit(
        X_train, y_train,
        X_test, y_test,
        5, batch_size
    )

    accuracies.append(score[1])

    mycnn.save_model("models/model_1_test.h5")

    y_pred_proba = mycnn.predict(X_test)
    y_pred = (y_pred_proba > 0.5) * 1
    print(classification_report(y_test, y_pred))
    area = plot_roc(y_test, y_pred_proba, title='ROC Curve CNN from scratch Epoch:' + str((i + 1) * 5))
    print("Area_ROC = {}".format(area))
    areas.append(area)

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.plot(accuracies, color = color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("AUC")
    ax2.plot(areas, color = color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Accuracy and AUC over time")
    plt.show()