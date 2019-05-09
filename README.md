# DeepEvents
A deep face recognition implementation using a deep convolutional neural network.

## Information
DeepEvents uses Keras with Tensorflow backend to implement the CNN, OpenCV and Dlib's landmark estimation to align faces.

The core idea behind this implementation is to be able to extract multiple people faces from images and generate a 128-dimensional embedding for each one of them. In this vector space, Euclidian distance shows as a good measure of face similarity.
Face classification is done by comparing embeddings with already labeled vectors. A SVM (Support Vector Machine) and a KNN classifier are trained to classify new images labels.
