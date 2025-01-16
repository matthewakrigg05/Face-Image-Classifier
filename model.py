import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from skimage import color
from skimage import io

Categories = ['AI', 'real']
flat_data_arr = []  # input array
target_arr = []  # output array
datadir = 'data/AI-face-detection-Dataset/'
# path which contains all the categories of images

for i in Categories:

    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

df = pd.DataFrame(flat_data)
df['Target'] = target

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

svc = svm.SVC(probability=True)
svc.fit(X=X_train, y=y_train)
y_pred = svc.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['AI', 'Real']))

accuracy = accuracy_score(y_pred, y_test)

# Print the accuracy of the model
print(f"The model is {accuracy * 100}% accurate")
