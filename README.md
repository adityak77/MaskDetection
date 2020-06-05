# MaskDetection

## About the project
This project leverages a CNN model that uses TensorFlow, OpenCV, Keras, and Scikit-learn to detect whether individuals in a picture are properly wearing masks. The mask detection model was created using the `train-model.py` file.

To detect masks using the `mask_detection.py` file, a two-step approach is used. Firstly, a face detection model delineates faces in the given image, and then the mask detection model places a rectangle around the face, indicating with certainty the presence of a mask.

## Usage
To run the model on any given image, run the following from the command line:
```
python mask_detection.py -f [path to face detector model] -m [path to mask detector model] -i [path to image]
```
If some of the arguments are not provided, a default path will be used.

## Sources
The face detection model that was used came from [www.pyimagesearch.com/](www.pyimagesearch.com) and the data that was used to train the mask detection model came from [https://www.kaggle.com/andrewmvd/face-mask-detection](https://www.kaggle.com/andrewmvd/face-mask-detection).