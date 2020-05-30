import xmltodict
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


DATASET_PATH = "dataset/667889-1176415-bundle-archive/"
OUTPUT_MODEL_PATH = "training_results/model75-25_SGD_final.h5"
OUTPUT_GRAPH_PATH = "training_results/graph75-25_SGD_final.jpeg"
NUM_IMAGES = 853
NUM_PROCESSED_IMAGES = 4072
IMAGE_SIZE = 64

# Hyperparamters
LEARN_RATE = 0.0001
EPOCHS = 50
BATCH_SIZE = 32

def load_labels(path):
    print('loading labels...')
    labels = []
    for i in range(NUM_IMAGES):
        f = xmltodict.parse(open(path + "annotations/maksssksksss{}.xml".format(i), 'rb'))
        if type(f['annotation']['object']) != list: # Only one mask
            f['annotation']['object'] = [f['annotation']['object']]

        lab = []
        for d in f['annotation']['object']:
            llist = [] # of the form [name, xmin, xmax, ymin, ymax]
            if d['name'] == 'without_mask':
                llist.append(0)
            elif d['name'] == 'mask_weared_incorrect':
                llist.append(1)
            else:
                llist.append(2)
            llist.append(int(d['bndbox']['xmin']))
            llist.append(int(d['bndbox']['xmax']))
            llist.append(int(d['bndbox']['ymin']))
            llist.append(int(d['bndbox']['ymax']))
            lab.append(llist)
        labels.append(lab)

    print('labels loaded...')
    return labels

def visualize(path, numImages):
    labels = load_labels(path)

    for i in range(numImages):
        lablist = labels[i]
        image = cv2.imread(path + "images/maksssksksss{}.png".format(i))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for lab in lablist:
            if lab[0] == 0:
                color = (255, 0, 0)
            elif lab[0] == 1:
                color = (255, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(image, (lab[1], lab[3]), (lab[2], lab[4]), color, 2)

        # plt.figure(figsize=(20, 20))
        # plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(image)
        plt.show()

# visualize(DATASET_PATH, 5)

def processImages(path):
    labels = load_labels(path)
    newLabels = []
    counter = 0
    for i in range(NUM_IMAGES):
        image = cv2.imread(path + "images/maksssksksss{}.png".format(i))
        # cv2.imshow("Image {}".format(i), image)
        # cv2.waitKey(0)

        label = labels[i]
        for llist in label:
            crop = image[(llist[3]):(llist[4]), (llist[1]):(llist[2])]
            # cv2.imshow("Cropped {}".format(i), crop)
            # cv2.waitKey(0)

            cv2.imwrite(path + "processed_images/image{}.png".format(counter), crop)
            counter += 1
            newLabels.append(llist[0])

    newLabels = np.asarray(newLabels)
    np.savetxt(path + "labels.csv", newLabels)

# processImages(DATASET_PATH)

def trainModel(path_to_data, output_model_path, output_graph_path):
    print("Retrieving data...")
    labels = np.genfromtxt(path_to_data + "labels.csv", delimiter=",")
    data = []
    for i in range(NUM_PROCESSED_IMAGES):
        image = load_img(path_to_data + "processed_images/image{}.png".format(i), target_size=(IMAGE_SIZE, IMAGE_SIZE))
        image = preprocess_input(img_to_array(image))
        data.append(image)

    labList = labels.tolist()
    newData = []
    newLabels = []
    cnt = 0
    for ind, classMask in enumerate(labList):
        if classMask == 0:
            newLabels.append(classMask)
            newData.append(data[ind])
        elif classMask == 2 and cnt < 800:
            newLabels.append(classMask)
            newData.append(data[ind])
            cnt += 1
    labels = np.array(newLabels)
    data = newData

    data = np.array(data, dtype="float")
    # print(Counter(labels))
    labels = np.reshape(labels, (-1, 1))
    oneHot = OneHotEncoder()
    labels = oneHot.fit_transform(labels).toarray()
    print("Creating Model...")

    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

    base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

    head = base.output
    head = AveragePooling2D(pool_size=(2, 2))(head)
    head = Flatten(name="flatten")(head)
    head = Dense(512, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    # head = Dense(3, activation="softmax")(head)
    head = Dense(2, activation="softmax")(head)

    model = Model(inputs=base.input, outputs=head)
    for layer in base.layers:
        layer.trainable = False

    print("Compiling Model...")

    # opt = Adam(lr=LEARN_RATE, decay= LEARN_RATE / EPOCHS)
    opt = SGD(lr=LEARN_RATE, momentum=0.9, decay= LEARN_RATE / EPOCHS)
    # model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("Training Model...")
    H = model.fit(aug.flow(trainX, trainY, batch_size=BATCH_SIZE), steps_per_epoch= len(trainX) // BATCH_SIZE, validation_data=(testX, testY),
                  validation_steps= len(testX) // BATCH_SIZE, epochs=EPOCHS)

    print("Evaluating Model...")
    predictions = model.predict(testX, batch_size=BATCH_SIZE)
    predictions = np.argmax(predictions, axis=1)
    # print(classification_report(testY.argmax(axis=1), predictions, target_names=["without_mask", "mask_weared_incorrect", "with_mask"]))
    print(classification_report(testY.argmax(axis=1), predictions, target_names=["without_mask", "with_mask"]))

    print("Saving model...")
    model.save(output_model_path, save_format="h5")

    print("Plotting model performance...")
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim((0, 1))
    plt.legend(loc="lower left")
    plt.savefig(output_graph_path)

trainModel(DATASET_PATH, OUTPUT_MODEL_PATH, OUTPUT_GRAPH_PATH)