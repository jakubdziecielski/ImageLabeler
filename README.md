# Image Classifier

## Introduction
The project contains:

* Python application for classifying pictures and saving results to csv file.
* Notebook to train the classification model (ResNet34 transfer learning).
* Trained model exported to .pkl

## Setting up the environment
To run the application a proper Python environment is required.
To set up the environment a prior installation of 
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is required.
Environment is set up by running the following command from application directory
```bash
conda env create -f environment.yml
```

Additionally, there is `environment.dev.yml` with packages used for development.
To install them run

```bash
conda env update --name image_labeler_env --file environment.dev.yml
```

## Running the application
To run the application first activate the environment and then run the main.py script.
Make sure to provide pictures to classify.
```bash
conda activate image_labeler_env
python main.py
```
The application has three optional arguments:

* `-i, --image_path` set on default to _input/_
* `-m, --model` set on default to _model.pkl_
* `-o, --output_path` set on default to _output/wynik.csv_

## Notebook remarks

Notebook was originally used in Google Colab. 
First section of cells is specifically meant to be run in Colab.
To run locally notebook should be run in the same environment as the application.

## Application in Docker
Attached Dockerfile can be used to build the image and run
the application in Docker. First make sure the images to classify are stored in `input/` directory 
in project root directory. 
To build the image from the project directory run
```bash
docker build -t image_labeler .
```
To run the image execute
```bash
docker run image_labeler
```
To copy the csv with classification results to current host location run
```bash
CONTAINER_ID=$(docker run -dit image_labeler)
docker cp $CONTAINER_ID:/app/output/wynik.csv .
```