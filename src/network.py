import caffe
import os
import numpy as np
import cv2
import paint_detection
import face_recognition

cifar_models_root = '/home/marta/PycharmProjects/facial_expression/models/'
cifar_weights_root = '/home/marta/PycharmProjects/facial_expression/results/'

# init model
model_config = os.path.join(cifar_models_root, 'deploy.prototxt')
model_weights = os.path.join(cifar_weights_root, '_iter_16000.caffemodel')
net = caffe.Net(model_config, model_weights, caffe.TEST)

face_cascade = cv2.CascadeClassifier('resource/haarcascade_frontalface_default.xml')
class_names = ["Angry", "Sad", "Happiness", "Surprised", "Neutral"]
colors = [(0, 0, 255), (100, 200, 100), (0, 255, 0), (255, 0, 255), (225, 0, 0)]
smiles = ['resource/images/angry.jpg', 'resource/images/sad.jpg', 'resource/images/happiness.jpg',
          'resource/images/surprised.jpg', 'resource/images/neutral.jpg']

j = 0
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    adapt_hist_equalization = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    faces = face_cascade.detectMultiScale(adapt_hist_equalization, 1.4, 10, minSize=(40, 40))
    # faces = face_cascade.detectMultiScale(cv2.equalizeHist(gray), 1.4, 8)

    for (x, y, w, h) in faces:
        j += 1
        roi_gr = gray[y: y + h, x: x + w]
        res = cv2.resize(roi_gr, (48, 48))
        img_new = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

        img_new = img_new.swapaxes(0, 2).swapaxes(1, 2)
        input_img = img_new.reshape((1, 3, 48, 48))
        input_img = input_img.astype(float) # convert each item from list to float
                                            # we cant use float(), because we cant transform list to float

        # predict
        net.blobs["data"].data[...] = input_img # input img for caffe
        prob = net.forward()['prob'].flatten()

        print 'Distribution:', prob, '\n'
        print 'Top-1: {0} ({1})'.format(class_names[np.argmax(prob)], np.max(prob))

        paint_detection.paint(frame, x, y, w, h, colors, prob, class_names, smiles)

    cv2.imshow('stream', frame)
    # cv2.imshow('faces', adapt_hist_equalization)
    # cv2.imshow('cv2.equalizeHist(gray)', cv2.equalizeHist(gray))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()