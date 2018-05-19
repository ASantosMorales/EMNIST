# EMNIST
First attemp to read handwritten numbers and letters using the EMNIST (Extension of the well known MNIST dataset https://www.nist.gov/itl/iad/image-group/emnist-dataset ).

## 1. Reading of dataset. [read_EMNIST_catalog.py]

The Matlab-format-dataset is used. First, the dataset is handled over Matlab/Octave due to its format. The extructures were managed and converted into matrises to an easy manipulation in Python.

In the next image the outcome of this first step is shown. The characters shown were chosen ramdomly.

![alt text](https://github.com/ASantosMorales/EMNIST/blob/master/EMNIST_illustration.png)

## 2. Characteristics extraction

## 2.1. Harris Corners. [Harris_corners.py]

In this section the Harris corners are gotten and the corresponding feature-vector is constructed.

The outcome getting is shown in the next image (somes examples).

![alt text](https://github.com/ASantosMorales/EMNIST/blob/master/Harris_corners.png)

## 2.2 Holes detection. [Holes_number.py]

It is asummed that the quantity of holes of a character is useful information. In this section the holes-number-vector is constructed. There are certains irregularities. For example to the zero character the expected holes-number is 1, but we have as outcome (in certain cases) 0 (when the zero is a non-closed shape), 2, 3 or more. 

In the next image it is evident why in some cases the expected outcome is not met.

![alt text](https://github.com/ASantosMorales/EMNIST/blob/master/Holes_number.png)

## 2.3 Elongation.

In work...
