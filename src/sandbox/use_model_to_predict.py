import sys

from skimage import io
from skimage.transform import resize

import numpy as np

from tensorflow.keras.models import load_model

model_path = sys.argv[1]

imgfile = sys.argv[2]

model = load_model(model_path)

X = []
img = io.imread(imgfile)
img = resize(img, (128,128))
X.append(img)

y_pred_proba = model.predict(np.array(X), verbose=1)
y_pred_proba = y_pred_proba[0][0]
if y_pred_proba >= 0.5:
    print("Malign, possibily is "+str(y_pred_proba))
if y_pred_proba < 0.5:
    print("Benign, possibily is "+str(1.0-y_pred_proba))

