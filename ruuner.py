import operator
import cv2 as cv
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog


# Functions
def findDescriptor(images):
    descriptors = {}
    for cls in images:
        descriptors[cls] = []
        for img in images[cls]:
            kp, des = orb.detectAndCompute(img,None)
            descriptors[cls].append(des)
    return descriptors


def findClassMatches(img, descriptors, threshold=15):
    kp2,des1 = orb.detectAndCompute(img,None)
    bf = cv.BFMatcher()
    matchList = {}
    for cls in descriptors:
        matchList[cls] = []
        for des in descriptors[cls]:
            matches = bf.knnMatch(des, des1, k=2)
            good = []
            for m,n in matches:
                if m.distance <0.75 *n.distance:
                    good.append([m])
            matchList[cls].append(len(good))

    classMatches = {}
    for cls in descriptors:
        if len(matchList[cls]) != 0:
            if max(matchList[cls]) > threshold:
                classMatches[cls] = max(matchList[cls])
    return classMatches


def bestMatch(classMatches):
    if len(classMatches) == 0:
        return False
    if len(classMatches) > 1:
        return max(classMatches.items(), key=operator.itemgetter(1))[0]
    else:
        key, value = classMatches.popitem()
        return key


# Declarations
train_folder = "instruments_data"
test_folder = "test_data"
images = {}
image_names = {}
classes = [
    "Acoustic",
    "Bass",
    "Drums",
    'Flute',
    "Gramophone",
    "Harp",
    "Piano",
    "Saxophone",
    "Tabla",
    "Violin"
]
orb = cv.ORB_create(nfeatures=1000)

# Get all image names in folders
for cls in classes:
    image_names[cls] = os.listdir(f'{train_folder}/{cls}')

# read all images to list
for cls in classes:
    images[cls] = []
    for name in image_names[cls]:
        img = cv.imread(f'{train_folder}/{cls}/{name}')
        images[cls].append(img)

# Get descriptors
descriptors = findDescriptor(images)

test_names = os.listdir(f'{test_folder}')
threshold_counter = []

for name in test_names:
    test_img = cv.imread(f'{test_folder}/{name}')
# while True:
#     root = tk.Tk()
#     root.withdraw()
#     test_img = cv.imread(filedialog.askopenfilename())
#     if test_img is None:
#         break

    matchedClasses = findClassMatches(test_img, descriptors)
    print(matchedClasses)
    matchedClass = bestMatch(matchedClasses)
    print(matchedClass)
    if matchedClass:
        threshold_counter.append(matchedClass)
        cv.putText(test_img, matchedClass, (0, 30), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        cv.imshow("test", test_img)
        cv.waitKey(0)
