import cv2
import dlib
import numpy as np
from pathlib import Path

def mask_lips(gray, padding=3, crop=True):
    model_path = ".\\models\\shape_predictor_68_face_landmarks_GTX.dat"
    model_path = Path(model_path).__str__()

    # Initialize the dlib face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    # Detect faces in the grayscale image
    faces = detector(gray)

    lip_img = np.zeros(gray.shape, dtype=np.uint8)
    for face in faces:
        # Detect the facial landmarks for the face
        landmarks = predictor(gray, face)

        # Create a bounding box around the lips
        lip_x = [landmarks.part(i).x for i in range(48, 68)]
        lip_y = [landmarks.part(i).y for i in range(48, 68)]

        if not lip_x or not lip_y:
            return None

        min_x, min_y = min(lip_x), min(lip_y)
        max_x, max_y = max(lip_x), max(lip_y)

        min_x, min_y = max(min_x - padding, 0), max(min_y - padding, 0)

        max_x, max_y = min(max_x + padding, gray.shape[1]), min(max_y + padding, gray.shape[0])

        lip_img[min_y:max_y, min_x:max_x] = gray[min_y:max_y, min_x:max_x]

    if crop:
        # Crop the image to the lip bounding box
        lip_img = lip_img[min_y:max_y, min_x:max_x]

    return lip_img

def mask_lips_batch(grays, padding=3, crop=True):
    model_path = ".\\models\\shape_predictor_68_face_landmarks_GTX.dat"
    model_path = Path(model_path).__str__()

    # Initialize the dlib face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    lip_imgs = []
    for gray in grays:
        # Detect faces in the grayscale image
        faces = detector(gray)

        lip_img = np.zeros(gray.shape, dtype=np.uint8)
        for face in faces:
            # Detect the facial landmarks for the face
            landmarks = predictor(gray, face)

            # Create a bounding box around the lips
            lip_x = [landmarks.part(i).x for i in range(48, 68)]
            lip_y = [landmarks.part(i).y for i in range(48, 68)]

            if not lip_x or not lip_y:
                continue

            min_x, min_y = min(lip_x), min(lip_y)
            max_x, max_y = max(lip_x), max(lip_y)

            min_x, min_y = max(min_x - padding, 0), max(min_y - padding, 0)

            max_x, max_y = min(max_x + padding, gray.shape[1]), min(max_y + padding, gray.shape[0])

            lip_img[min_y:max_y, min_x:max_x] = gray[min_y:max_y, min_x:max_x]

        if crop:
            # Crop the image to the lip bounding box
            lip_img = lip_img[min_y:max_y, min_x:max_x]

        lip_imgs.append(lip_img)

    return lip_imgs


def img2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def scale(img, output_size):
    return cv2.resize(img, output_size)

def edge_detection(img):
    return cv2.Canny(img, 10, 10)