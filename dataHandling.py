import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_directories():
    flat_data_arr = []  # input array
    target_arr = []  # output array

    for directory in ['data/AI', 'data/real']:
        for img in os.listdir(directory):
            img_array = imread(os.path.join(directory, img))
            img_resized = resize(img_array, (150, 150, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(directory)

    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)

    df = pd.DataFrame(flat_data)
    df['Target'] = target

    return df


def split_data(dataframe):
    x = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test