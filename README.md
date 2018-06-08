# EMNIST
First attemp to read handwritten numbers and letters using the EMNIST (Extension of the well known MNIST dataset https://www.nist.gov/itl/iad/image-group/emnist-dataset ).

## 1. Reading of dataset. [read_EMNIST_catalog.py]

The Matlab-format-dataset is used. First, the dataset is handled over Matlab/Octave due to its format. The extructures were managed and converted into matrises to an easy manipulation in Python.

In the next image the outcome of this first step is shown. The characters shown were chosen ramdomly.

![alt text](https://github.com/ASantosMorales/EMNIST/blob/master/EMNIST_illustration.png)

## 2. Characteristics extraction

#### 2.1. Harris Corners. [Harris_corners.py]

In this section the Harris corners are gotten and the corresponding feature-vector is constructed.

The outcome getting is shown in the next image (somes examples).

![alt text](https://github.com/ASantosMorales/EMNIST/blob/master/Harris_corners.png)

#### 2.2 Holes detection. [Holes_number.py]

It is asummed that the quantity of holes of a character is useful information. In this section the holes-number-vector is constructed. There are certains irregularities. For example to the zero character the expected holes-number is 1, but we have as outcome (in certain cases) 0 (when the zero is a non-closed shape), 2, 3 or more. 

In the next image it is evident why in some cases the expected outcome is not met.

![alt text](https://github.com/ASantosMorales/EMNIST/blob/master/Holes_number.png)

#### 2.3 Elongation. [Elongation.py]

The elongation is defined (in this work) as the ratio between the mayor axis and the minor axis of the ellipse that encloses a certain character.

The following image shows some random examples of how the feature-vector was constructed.

![alt text](https://github.com/ASantosMorales/EMNIST/blob/master/Elongation.png)

#### 2.4 Histogram of Oriented Gradients (HOG)

The HOG description is getting in this section. The number of bins is 12 (i.e. pi/6, pi/3, ..., 2pi) and the image is divided by 4 quadrants to perfom the analysis. In the next image is shown the HOG description of the three number.

<p align="center">
<img src="https://github.com/ASantosMorales/EMNIST/blob/master/HOG_description.jpeg">
</p>

![alt text](https://github.com/ASantosMorales/EMNIST/blob/master/HOG_description.jpeg)

Finally the outcome feature-vector is a vector [1 x 48] for each image (12 bins x 4 quadrants).

Support link: https://docs.opencv.org/3.1.0/dd/d3b/tutorial_py_svm_opencv.html (In this link a deeply explanation can be found)

## 3. Characters determination

#### 3.1 Support vector machine. In work...
