from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', default='face_detector/', help='path to directory with face detector model')
ap.add_argument('-m', '--model', default='training_results/model75-25_Adam_balanced_dataset.h5', help='path to file with mask detector model')
ap.add_argument('-i', '--image', default='examples/example_03.jpg', help='path to file with image')
args = vars(ap.parse_args())

FACE_DETECTOR_PATH = args['face']
MODEL_PATH = args['model']
EXAMPLE_PATH = args['image']

def detectMask(path_to_face_detector_model, path_to_mask_model, path_to_image):
    print("Loading models...")
    face_model = cv2.dnn.readNet(path_to_face_detector_model + "deploy.prototxt", path_to_face_detector_model + "res10_300x300_ssd_iter_140000.caffemodel")
    mask_model = load_model(path_to_mask_model)
    image = cv2.imread(path_to_image)
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    print("Detecting faces...")
    face_model.setInput(blob)
    detections = face_model.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            startX, startY, endX, endY = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(endX, width - 1)
            endY = min(endY, height - 1)

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (64, 64))
            face = np.expand_dims(preprocess_input(img_to_array(face)), axis=0)

            withoutMask, withMask = mask_model.predict(face)[0]
            if withMask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(withMask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Output", image)
    cv2.waitKey(0)

detectMask(FACE_DETECTOR_PATH, MODEL_PATH, EXAMPLE_PATH)