# DeepEvents
A deep face recognition implementation using a deep convolutional neural network.

## Information
DeepEvents uses Keras with Tensorflow backend to implement the CNN, OpenCV and Dlib's landmark estimation to align faces. Inspired by Google's FaceNet and OpenFace

The core idea behind this implementation is to be able to extract multiple people faces from images and generate a 128-dimensional embedding for each one of them. In this vector space, Euclidian distance shows as a good measure of face similarity.
Face classification is done by comparing embeddings with already labeled vectors. A SVM (Support Vector Machine) and a KNN classifier are trained to classify new images labels.

## Install

In order to run this code it is needed to have a jupyter notebook + tensorflow + opencv + keras running. It's easier use docker, Dockerfile is provided to build it.

Once the container is all set up, run:
```
$ mkdir models && cd models
$ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
$ bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
$ mv shape_predictor_68_face_landmarks.dat landmarks.dat
