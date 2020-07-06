from itertools import chain

import inline
import matplotlib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_metadata(data_folder):

    metadata = pd.read_csv(os.path.join(data_folder, 'metadata\chest_xray_metadata.csv'))
    modifiedDataset = metadata.fillna(" ")
    file_system_scan_jpeg = {os.path.basename(x): x for x in
                         glob(os.path.join('.', data_folder, '*.jpeg'))}
    file_system_scan_jpg = {os.path.basename(x): x for x in
                        glob(os.path.join('.', data_folder, '*.jpg'))}
    file_system_scan_png = {os.path.basename(x): x for x in
                        glob(os.path.join('.', data_folder, '*.png'))}
    print('Scans found:', len(file_system_scan_jpeg), len(file_system_scan_jpg), len(file_system_scan_png))
    file_system_scan = {**file_system_scan_jpeg, **file_system_scan_jpg,  **file_system_scan_png}
    modifiedDataset['path'] = modifiedDataset['X_ray_image_name'].map(file_system_scan.get)

    print( modifiedDataset['path'])
    print('Total x-ray records:{}.'.format((metadata.shape[0])))
    return modifiedDataset

def preprocess_metadata(metadata):
    # izbaciti smoking Stress-Smoking
    # ici po label1 razdvajanje Label_1_Virus_category

    print(metadata['Label_1_Virus_category'])
    metadata['Label_1_Virus_category'] = metadata['Label_1_Virus_category'].map(
        lambda x: x.replace(' ', 'Normal'))

    print(metadata['Label_1_Virus_category'])

    labels = np.unique(
        list(chain(*metadata['Label_1_Virus_category'].map(lambda x: x.split('|')).tolist())))
    labels = [x for x in labels if len(x) > 0]

    for c_label in labels:
        if len(c_label) > 1:  # leave out empty labels
            metadata[c_label] = metadata['Label_1_Virus_category'].map(
                lambda finding: 1.0 if c_label in finding else 0)

    #odbaci pusenje
    labels = [c_label for c_label in labels if metadata[c_label].sum() > 2]

    sample_weights = metadata['Label_1_Virus_category'].map(
        lambda x: len(x.split('|')) if len(x) > 0 else 0).values + 4e-2
    sample_weights /= sample_weights.sum()


    labels_count = [(c_label, int(metadata[c_label].sum()))
                    for c_label in labels]

    print('Labels ({}:{})'.format((len(labels)), (labels_count)))
    print('Total x-ray records:{}.'.format((metadata.shape[0])))

    return metadata, labels


def stratify_train_test_split(metadata):

    stratify = metadata['Label_1_Virus_category'].map(lambda x: x[:4])
    train, valid = train_test_split(metadata,
                                    test_size=0.25,
                                    random_state=2018,
                                    stratify=stratify)
    print('train', train.shape[0], 'validation', valid.shape[0])
    return train, valid

#if __name__ == '__main__':


data_folder = 'data\chest_xray_data_set\chest_xray_data_set'

metadata = load_metadata(data_folder)
metadata, labels = preprocess_metadata(metadata)
metadata['disease_vec'] = metadata.apply(lambda x: [x[labels].values], 1).map(lambda x: x[0])
train, valid = stratify_train_test_split(metadata)

train.fillna(False)
valid.fillna(False)

