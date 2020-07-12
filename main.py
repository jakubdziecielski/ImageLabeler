"""
ImageLabeler application
"""

import os
from argparse import ArgumentParser
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from fastai.basic_train import DatasetType, Learner, load_learner
from fastai.vision import ImageList, ItemList
from torch import tensor

ap = ArgumentParser(description="Generate csv with picture labels, classify as either Dog, Cat or Human.")

ap.add_argument(
    '-i',
    '--image_path',
    default='input' + os.path.sep,
    help='Path to location of images to label.'
)

ap.add_argument(
    '-m',
    '--model_path',
    default='model.pkl',
    help='Path to classifier.'
)

ap.add_argument(
    '-o',
    '--output_path',
    default='output' + os.path.sep + 'wynik.csv',
    help='Location to save the results of labelling.'
)

args = ap.parse_args()


def get_data(data_path: str) -> Union[ImageList, ItemList]:
    """
    Loads images from specified location
    :param data_path: Path to location with images
    :return: Object that can be loaded into learner
    """

    assert os.path.exists(data_path), 'Specified input location not found, stopping the application.'
    return ImageList.from_folder(data_path)


def get_learner(model_path: str, data: ImageList) -> Learner:
    """
    Loads classification model
    :param model_path: Model (.pkl) location
    :param data: Data to load into learner
    :return: fastai Learner object
    """
    return load_learner(
        path=os.path.sep.join(model_path.split(os.path.sep)[:-1]),
        file=model_path.split(os.path.sep)[-1],
        test=data
    )


def label_pictures():
    """
    Main function of this application. Gets the pictures, gets the model, classifies the pictures and saves the
    classification results to csv file.
    """
    data: ImageList = get_data(args.image_path)
    assert len(data.items), 'No images found, stopping the application.'

    learner: Learner = get_learner(args.model_path, data)

    classes: List[str] = learner.data.classes
    class_mapping: Dict[str, int] = {name: i for i, name in enumerate(classes)}

    preds, _ = learner.get_preds(ds_type=DatasetType.Test)

    labels: tensor = np.argmax(preds, 1)
    sparse_labels = [[0 for _ in labels] for _ in classes]
    for i, label in enumerate(labels):
        sparse_labels[label][i] = 1

    image_num = len(learner.data.test_ds)
    file_names = []
    for i in range(image_num):
        file_names.append(str(learner.data.test_ds.items[i]).split(os.path.sep)[-1])

    output_dir = ''.join(args.output_path.split(os.path.sep)[:-1])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pd.DataFrame(
        data={
            'file_name': file_names,
            'dog': sparse_labels[class_mapping['dog']],
            'cat': sparse_labels[class_mapping['cat']],
            'human': sparse_labels[class_mapping['human']]
        }
    ).to_csv(args.output_path, index=False)


if __name__ == '__main__':
    label_pictures()

# TODO: Add colab notebook
