# animal_classifier
Classifying animals with TensorFlow python!

## Introduction

What is an image classifier? It is a machine learning model which can classify an image. You’ll show it an image, and it’ll tell you what it thinks it is. The development process of our project includes building an image dataset of 10 animals, designing the UI using Django Web Framework, loading the image dataset, building the image classifier model, training and validating the model, saving and loading trained model’s weights, making predictions using the trained model, and experimental results.

## Building Dataset

The dataset holds all the images. A separate directory should be created for each of the following:

* Training Dataset: images which are used during model training
* Validation Dataset: images which are used during model validation 

Each image dataset should be organized as directories of images. They should be named by the class (e.g. cat) of images it holds. We have created 10 classes with 10 images of each. These animals include butterfly, cat, chicken, giraffe, penguin, snake, spider, tiger, wolf, zebra. It is important that none of the images in the training dataset is in the validation dataset.

![Dataset Folders](/screenshots/dataset_folders.png)

## Designing the UI with Django web framework

Django automatically creates all necessary files like manage.py, urls.py, settings.py, views.py and few others when we start the project. Manage.py helps us to run our project in the localhost server. Settings.py help us to connect our project components between each other, for example directory of html files, media file directories and connection of database in our project. Urls.py manages all requests from html and browser. Views.py contains codes to process the request that it received from urls.py.

For our project we will work in views.py and html files. Using html we will construct the design of our UI. In the html we will contain: forms, train button, input file, test button, image and text. When train button is clicked, form will send request to url.py and through url.py views.py will receive the request and the training code will run. After training is finished we can chose the image and press test button. Test button will send request and view.py will return the classification result. 

## Training & Validating the Model

![Training](/screenshots/training.png)
![Training Result](/screenshots/training_finished.png)

## Make Predictions Using the Trained Model, Experimental Results

![Training](/screenshots/result.png)