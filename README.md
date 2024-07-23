# XML_Augment

## Overview

`XML_Augment` is a Python library for augmenting images and their corresponding XML annotations, typically used in object detection and image segmentation tasks. It includes functionalities for visualizing annotations, brightening images, translating images, and rotating images while updating the XML annotations accordingly.

## Features

- **Visualize Annotations**: Display the annotated polygons and bounding boxes on the image.
- **Brighten Images**: Randomly adjust the brightness of the image.
- **Translate Images**: Randomly translate (shift) the image and update the annotations.
- **Rotate Images**: Randomly rotate the image and update the annotations.

## Installation

To use this project, ensure you have the following libraries installed:

```bash
pip install xml_augmenter
```
## Usage

### Requirements

- Python 3.x
- OpenCV
- NumPy

##Example

#### Import necessary libraries
import cv2 as cv
import xml.etree.ElementTree as ET
import numpy as np

#### Define the path to your image and XML annotation file
impath = "path/to/your/image.jpg"
anpath = "path/to/your/annotations.xml"

#### Create an instance of XML_Augment
xa = XML_Augment(anpath, impath)

#### Visualize annotations with custom text size
xa.visualize_annotaitons(text_size=0.3)

#### Apply translation augmentation with a maximum translation percent
xa.translate(0.2)

#### Apply brightness augmentation with a maximum brightness value
xa.brighten(100)

#### Apply rotation augmentation with a maximum rotation offset
xa.rotate(45)

#### Visualize annotations again after augmentations
xa.visualize_annotaitons(text_size=0.3)

#### Show the final image with augmentations
cv.imshow('image', xa.image)
cv.waitKey()
cv.destroyAllWindows()

