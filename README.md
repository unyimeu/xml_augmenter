# XML_Augment

## Overview

`XML_Augment` is a Python library for augmenting images and their corresponding XML annotations, typically used in object detection and image segmentation tasks. It includes functionalities for visualizing annotations, brightening images, translating images, and rotating images while updating the XML annotations accordingly.

## Features

- **Visualize Annotations**: Display the annotated polygons and bounding boxes on the image.
- **Brighten Images**: Randomly adjust the brightness of the image.
- **Translate Images**: Randomly translate (shift) the image and update the annotations.
- **Rotate Images**: Randomly rotate the image and update the annotations.[ROTATION NOT SUPPORTED YET VERSION<=0.0.2]

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

## Example

#### Import necessary libraries
```bash
import cv2 as cv
import xml.etree.ElementTree as ET
import numpy as np
```
#### Define the path to your image and XML annotation file
```bash
impath = "path/to/your/image.jpg"
anpath = "path/to/your/annotations.xml"
```
#### Create an instance of XML_Augment
```bash
xa = XML_Augment(anpath, impath)
```
#### Visualize annotations with custom text size
```bash
xa.visualize_annotaitons(text_size=0.3)
```
#### Apply translation augmentation with a maximum translation percent
```bash
xa.translate(0.2)
```
#### Apply brightness augmentation with a maximum brightness value
```bash
xa.brighten(100)
```
#### Apply rotation augmentation with a maximum rotation offset [ROTATION NOT SUPPORTED AS OF VERSION 0.0.2]
```bash
xa.rotate(45)
```
#### Visualize annotations again after augmentations
```bash
xa.visualize_annotaitons(text_size=0.3)
```
#### Show the final image with augmentations
```bash
cv.imshow('image', xa.image)
cv.waitKey()
cv.destroyAllWindows()
```

