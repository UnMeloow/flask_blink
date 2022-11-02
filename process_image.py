import cv2
import dlib
import numpy as np
from imutils import face_utils


class Colors:
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    VIOLET = (255, 0, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    LIGHT_GREY = (128, 128, 128)
    DARK_GREY = (60, 60, 60)


predictor = dlib.shape_predictor('./static/shape_predictor_68_face_landmarks.dat')
face_cascade = cv2.CascadeClassifier('./static/haarcascade_frontalface_alt.xml')


# detect the face rectangle
def detect(img, cascade=face_cascade, minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)

    # if it doesn't return rectangle return array
    # with zero lenght
    if len(rects) == 0:
        return []

    #  convert last coord from (width,height) to (maxX, maxY)
    rects[:, 2:] += rects[:, :2]

    return rects


def cnnPreprocess(img):
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img


def cropEyes(frame, im_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect the face at grayscale image
    te = detect(gray, minimumFeatureSize=(80, 80))

    # if the face detector doesn't detect face
    # return None, else if detects more than one faces
    # keep the bigger and if it is only one keep one dim
    if len(te) == 0:
        return None
    elif len(te) > 1:
        face = te[0]
    elif len(te) == 1:
        [face] = te

    # keep the face region from the whole frame
    face_rect = dlib.rectangle(left=int(face[0]), top=int(face[1]),
                               right=int(face[2]), bottom=int(face[3]))

    # determine the facial landmarks for the face region
    shape = predictor(gray, face_rect)
    shape = face_utils.shape_to_np(shape)

    #  grab the indexes of the facial landmarks for the left and
    #  right eye, respectively
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # extract the left and right eye coordinates
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    # keep the upper and the lower limit of the eye
    # and compute the height
    l_uppery = min(leftEye[1:3, 1])
    l_lowy = max(leftEye[4:, 1])
    l_dify = abs(l_uppery - l_lowy)

    # compute the width of the eye
    lw = (leftEye[3][0] - leftEye[0][0])

    minxl = (leftEye[0][0] - ((im_size - lw) / 2))
    maxxl = (leftEye[3][0] + ((im_size - lw) / 2))
    minyl = (l_uppery - ((im_size - l_dify) / 2))
    maxyl = (l_lowy + ((im_size - l_dify) / 2))

    # crop the eye rectangle from the frame
    left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
    left_eye_rect = left_eye_rect.astype(int)
    left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]

    # same as left eye at right eye
    r_uppery = min(rightEye[1:3, 1])
    r_lowy = max(rightEye[4:, 1])
    r_dify = abs(r_uppery - r_lowy)
    rw = (rightEye[3][0] - rightEye[0][0])
    minxr = (rightEye[0][0] - ((im_size - rw) / 2))
    maxxr = (rightEye[3][0] + ((im_size - rw) / 2))
    minyr = (r_uppery - ((im_size - r_dify) / 2))
    maxyr = (r_lowy + ((im_size - r_dify) / 2))
    right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
    right_eye_rect = right_eye_rect.astype(int)
    right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

    # if it doesn't detect left or right eye return None

    if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
        return None
    # resize for the conv net
    left_eye_image = cv2.resize(left_eye_image, (im_size, im_size))
    right_eye_image = cv2.resize(right_eye_image, (im_size, im_size))
    left_eye_image = cv2.flip(right_eye_image, 1)
    # return left and right eye
    return left_eye_image, right_eye_image


def classify_image(image_path, eye_model, im_size):
    image = cv2.imread(image_path)
    left_eye_res = "Ваш левый глаз: "
    right_eye_res = "Ваш правый глаз: "

    try:
        left_eye, right_eye = cropEyes(image, im_size)
    except TypeError:
        left_eye_res += "не удалось определить."
        right_eye_res += "не удалось определить."
        return left_eye_res, right_eye_res

    left_eye = cnnPreprocess(left_eye)
    right_eye = cnnPreprocess(right_eye)
    left_eye_pred = eye_model.predict(left_eye)
    right_eye_pred = eye_model.predict(right_eye)

    if left_eye_pred <= 0.1:
        left_eye_res += "закрыт."
    else:
        left_eye_res += "открыт."
    if right_eye_pred <= 0.1:
        right_eye_res += "закрыт."
    else:
        right_eye_res += "открыт."

    return left_eye_res, right_eye_res
