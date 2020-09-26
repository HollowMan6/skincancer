import sys

from tensorflow.keras.models import load_model

from ml_lib.moleimages import MoleImages

from ml_lib.roc import classification_report, plot_roc

model_path = sys.argv[1]

model = load_model(model_path)

mimg = MoleImages()
X_test, y_test = mimg.load_test_images('data_scaled_validation/benign', 'data_scaled_validation/malign')

X_train, y_train = mimg.load_test_images('data_scaled/benign', 'data_scaled/malign')

print("Test Dataset")
y_pred_proba = model.predict(X_test, verbose=1)
y_pred = (y_pred_proba > 0.5) * 1
print(classification_report(y_test, y_pred))
area = plot_roc(y_test, y_pred_proba,
                title="")
print("Area Under ROC = {}".format(area))

print("Train Dataset")
y_pred_proba = model.predict(X_train, verbose=1)
y_pred = (y_pred_proba > 0.5) * 1
print(classification_report(y_train, y_pred))
area = plot_roc(y_train, y_pred_proba,
                title="")
print("Area Under ROC = {}".format(area))
