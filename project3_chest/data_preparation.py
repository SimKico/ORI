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

    # data_folder = 'data\chest_xray_data_set\chest_xray_data_set\metadata\chest_xray_metadata.csv'

    metadata = pd.read_csv(os.path.join(data_folder))
    modifiedDataset = metadata.fillna(" ")
    file_system_scan = {os.path.basename(x): x for x in
                        glob(os.path.join('.', 'data', 'images*', '*.png'))}
    # if len(file_system_scan) != metadata.shape[0]:
    #     raise Exception(
    #         'ERROR: Different number metadata records and png files.'.format())

    modifiedDataset['path'] = modifiedDataset['X_ray_image_name'].map(file_system_scan.get)
    print('Total x-ray records:{}.'.format((metadata.shape[0])))

    return modifiedDataset

# all_xray= pd.read_csv('data\chest_xray_data_set\chest_xray_data_set\metadata\chest_xray_metadata.csv')
# all_image_paths = {os.path.basename(x): x for x in
#                    glob(os.path.join('.', 'data', 'images*', '*.png'))}
# print('Scans found:', len(all_image_paths), ', Total Headers', all_xray.shape[0])
# all_xray['path'] = all_xray['X_ray_image_name'].map(all_image_paths.get)
# all_xray['path'] = all_xray['X_ray_image_name'].map(all_image_paths.get)

# print(all_xray.sample(10))
#
# label_counts = all_xray['Label'].value_counts()[:15]
# fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
# ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
# ax1.set_xticks(np.arange(len(label_counts))+0.5)
# _ = ax1.set_xticklabels(label_counts.index, rotation = 90)

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

    labels = [c_label for c_label in labels if metadata[c_label].sum()
              > 2]

    sample_weights = metadata['Label_1_Virus_category'].map(
        lambda x: len(x.split('|')) if len(x) > 0 else 0).values + 4e-2
    sample_weights /= sample_weights.sum()

    # metadata = metadata.sample(5286, weights=sample_weights)

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
    return train, valid

if __name__ == '__main__':
    data_folder = 'data\chest_xray_data_set\chest_xray_data_set\metadata\chest_xray_metadata.csv'

    metadata = load_metadata(data_folder)
    metadata, labels = preprocess_metadata(metadata)
    train, valid = stratify_train_test_split(metadata)