# EMNIST
First attemp to read handwritten numbers and letters using the EMNIST (Extension of the well known MNIST dataset https://www.nist.gov/itl/iad/image-group/emnist-dataset ).

1. Reading of dataset.
The Matlab-format-dataset is used. First, the dataset is handled over Matlab/Octave due to its format. The extructures were managed and converted into matrises to an easy manipulation in Python.

In the next image the outcome of this first step is shown. The characters shown were chosen ramdomly.

![alt text](https://github.com/ASantosMorales/EMNIST/blob/master/EMNIST_illustration.png)


2. Characteristics extraction

2.1. Harris Corners.
In this section the Harris corners are gotten and the corresponding feature-vector is constructed.
script_name: Harris_corners.py
The outcome getting is shown in the next image (somes examples).

![alt text](https://github.com/ASantosMorales/EMNIST/blob/master/Harris_corners.png)
