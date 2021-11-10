# File paths
train_json_path = 'PennFudanPed_train.json'
test_json_path = 'PennFudanPed_val.json'
output_path = 'preds.json'

import cv2
import glob
import os
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
import numpy as np
import json

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

f = open(test_json_path, 'r')
test_data = json.load(f)

predictions = []

for img_data in test_data['images']:
    print('Looking up image', img_data['id'])
    img = cv2.imread(img_data['file_name'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(img, winStride=(4, 4),padding=(8, 8),scale=1.03) # after hyper parameter tuning

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.3)

    for rect in pick:
        bbox = list(map(float, [rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]]))
        predictions.append({
                        'image_id':img_data['id'],
                        'category_id':1,
                        'bbox':bbox,
                        'score':1.0
                    })
        
    # for (x, y, w, h) in pick:
    #     cv2.rectangle(img,(x, y),(w, h),(0, 255, 0),2)
  
    # plt.imshow(img)


f.close()

with open(output_path, 'w') as f_out:
    json.dump(predictions , f_out)