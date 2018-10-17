# Developing an AI application for flower classification

- In this project I train an image classifier to recognize different species of flowers among 102 flower categories. 
- Use the trained classifier to predict image content of top-K predictions
- Develope a Command Line Application to allows users to set hyperparameters for learning rate, number of hidden units, training epochs, GPU/CPU

![Flowers](/assets/Flowers.png)


# Dependecies and packages:

- Python 3.x
- Numpy
- PyTorch
- PIL
- json
- matplotlib
- collections


# Repository content:

- Jupyter notebook file: `Image Classifier Project_july01_2018_submit`
- html file of Jupyter notebook: `Image Classifier Project_july01_2018_submit.html`
- Image folder (`assets`): containing figures in the notebook
- A json file: `cat_to_name.json` containing a dictionary that translate folder name to flower name
- A helper file: `workspace_utils.py` containing `active_session` function that used to avoid disconnecting program during long run
- Two Python file: `train_submission.py` and `predict_submission.py` for train and prediction using command line for Image classifier
- Two txt file : `sample_train_in_out.txt` and `sample_predict_in_out.txt` containg a successful printout of `train.py` and `predict.py` in the output
- MIT License file


# Basic Usage for command line

- Clone the repository use: `git clone https://github.com/ania4data/Developing_AI_application_for_classification.git`

## To train 

Train the network on the dataset

- Basic usage: python `train_submission.py` data_directory
- Prints out training loss, validation loss, and validation accuracy as the network trains
- Set directory to save checkpoints: python `train_submission.py` data_dir --save_dir save_directory
- Choose architecture: python `train_submission.py` data_dir --arch "resnet152"
- Set hyperparameters: python `train_submission.py` data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use GPU for training: python `train_submission.py` data_dir --gpu

## To predict

Predict flower name from an image with `predict_submission.py` along with the probability of that name. 

- Basic usage: python `predict_submission.py` /path/to/image checkpoint_from_training
- Return top K most likely classes: python `predict_submission.py` input checkpoint_from_training --top_k 3
- Use a mapping of categories to real names: python `predict_submission.py` input checkpoint_from_training --category_names cat_to_name.json
- Use GPU for inference: python `predict_submission.py` input checkpoint_from_training --gpu


# Source dataset

The zipped `102flowers.tgz` file can be downloaded from download section of Visual Geometry Group in Oxford University website:

- [Link to the study](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

