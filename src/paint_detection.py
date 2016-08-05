import cv2
import numpy as np

def paint(frame, x, y, w, h, colors, prob, class_names, smiles):
    cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), colors[np.argmax(prob)], -1)
    cv2.putText(frame, class_names[np.argmax(prob)], (x, y + h + 23), cv2.FONT_ITALIC, h / 180.0, (255, 255, 255), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colors[np.argmax(prob)], 1)
    img = cv2.imread(smiles[np.argmax(prob)])
    img = cv2.resize(img, (40, 40))
    rows, cols, channels = img.shape
    try:
        frame[y: y + cols, x + w: x + w + rows] = img
    except:
        pass