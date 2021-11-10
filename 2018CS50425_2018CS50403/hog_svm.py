# File paths
train_json_path = '../PennFudanPed_train.json'
test_json_path = '../PennFudanPed_val.json'
output_path = '../preds.json'
root_dir = '../../pedestrian_detection/'

import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import json
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Window parameters
window_size = (64, 128)
window_stride = 10

# hog paramaters
convert_to_grayscale = False
orientations=9
pixels_per_cell=(8, 8)
cells_per_block=(3, 3)
block_norm='L2-Hys'
transform_sqrt=True

# Gaussian pyramid parameters
pyramid_downscale=1.5

# score threshold
score_thresh = 0.8

# Non maximal suppression parameters
nms_thresh=0.2

# SVM parameters
tol = 1e-4
C = 1.0

# Negative sampling
num_neg = 20
iou_thresh = 0.3

# Random state
random_state = 42
np.random.seed(random_state)


def iou(bb1,bb2):
    bb1 = (bb1[0], bb1[1], bb1[2] + bb1[0], bb1[3] + bb1[1])
    bb2 = (bb2[0], bb2[1], bb2[2] + bb2[0], bb2[3] + bb2[1])

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    if x_right < x_left or y_bottom < y_top:
        return 0

    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    if iou < 0 or iou > 1:
        iou = 0

    return iou

# taken from imutils library and modified to return scores too
# (from https://github.com/PyImageSearch/imutils/blob/master/imutils/object_detection.py)
def non_max_suppression_with_scores(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int"), np.array(probs)[pick]

def prepare_dataset():
    image_paths = {}
    images_bboxes = {}

    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

        for img in train_data['images']:
            image_paths[img['id']] = img['file_name']
        for ann in train_data['annotations']:
            if ann['category_id']==1 and ann['ignore']==0:
                img_id = ann['image_id']
                if img_id not in images_bboxes:
                    images_bboxes[img_id] = []
                images_bboxes[img_id].append(ann['bbox'])

    x_positive = []
    x_negative = []

    for img_id in image_paths:
        img = cv2.imread(os.path.join(root_dir, image_paths[img_id]))
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Positive samples
        for bbox in images_bboxes[img_id]:
            x, y, w, h = map(int, bbox)
            img_p = img[y:y+h, x:x+w]
            img_p = cv2.resize(img_p, window_size)
            x_positive.append(img_p)

        # Negative samples
        # Try data augmentation, diff window sizes, pyramid etc
        if img.shape[1] > window_size[0] and img.shape[0] > window_size[1]:
            for i in range(num_neg):
                x = np.random.randint(0, img.shape[1] - window_size[0])
                y = np.random.randint(0, img.shape[0] - window_size[1])
                non_overlapping = True
                for bbox in images_bboxes[img_id]:
                    if iou((x, y, window_size[0], window_size[1]), bbox) > iou_thresh:
                        non_overlapping = False
                        break
                if non_overlapping:
                    img_n = img[y:y+window_size[1], x:x+window_size[0]]
                    x_negative.append(img_n)

    print('Positive samples:', len(x_positive))
    print('Negative samples:', len(x_negative))

    x_images = x_positive + x_negative
    y = [1 for i in range(len(x_positive))] + [0 for i in range(len(x_negative))]
    x = []

    for i, img in enumerate(x_images):
        if convert_to_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        desc = hog(img, feature_vector=True, multichannel=not(convert_to_grayscale),
                    orientations=orientations, pixels_per_cell= pixels_per_cell, cells_per_block= cells_per_block,
                    block_norm=block_norm, transform_sqrt=transform_sqrt)
        x.append(desc)

    x = np.array(x)
    y = np.array(y)
    x, y = shuffle(x, y, random_state=random_state)

    return x, y


def train_model(x, y):
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    model = LinearSVC(tol=tol, C=C, random_state = random_state)
    model.fit(x, y)
    return scaler, model

def get_image_predictions(image_id, file_name, model, scaler): 
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if convert_to_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bboxes = []
    scores = []

    cur_scale = 1.0

    for img in pyramid_gaussian(img, downscale=pyramid_downscale, multichannel=not(convert_to_grayscale)):
        for x_s in range(0, img.shape[1] - window_size[0], window_stride):
            for y_s in range(0, img.shape[0] - window_size[1], window_stride):
                img_candidate = img[y_s:y_s+window_size[1], x_s:x_s+window_size[0]]
                desc = hog(img_candidate, feature_vector=True, multichannel=not(convert_to_grayscale),
                            orientations=orientations, pixels_per_cell= pixels_per_cell, cells_per_block= cells_per_block,
                            block_norm=block_norm, transform_sqrt=transform_sqrt)
                desc = scaler.transform([desc])
                pred = model.predict(desc)
                score = model.decision_function(desc)[0]

                if pred==1 and score > score_thresh:
                    bboxes.append([x_s * cur_scale, y_s * cur_scale, window_size[0] * cur_scale, window_size[1] * cur_scale])
                    scores.append(score)

        cur_scale = cur_scale * pyramid_downscale

    rects = np.array([(x, y, x+w, y+h) for (x, y, w, h) in bboxes])
    try:
        rects, scores = non_max_suppression_with_scores(rects, probs = scores, overlapThresh=nms_thresh)
    except:
        return []
    # try out tf.image.non_max_suppression_with_scores to get scores too?

    predictions = []
    for i, rect in enumerate(rects):
        bbox = list(map(float, [rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]]))
        predictions.append({
                        'image_id':image_id,
                        'category_id':1,
                        'bbox':bbox,
                        'score':scores[i]
                    })
    return predictions


def get_and_write_predictions(scaler, model):
    f = open(test_json_path, 'r')
    test_data = json.load(f)

    predictions = []

    for img_data in test_data['images']:
        print('Looking up image', img_data['id'])
        img_predictions = get_image_predictions(img_data['id'], img_data['file_name'], model, scaler)
        predictions += img_predictions
    f.close()

    print('Writing predictions to file..')

    with open(output_path, 'w') as f_out:
        json.dump(predictions , f_out)

    print('Written predictions to', output_path)

    return predictions


x, y = prepare_dataset()

scaler, model = train_model(x, y)

import pickle
with open("model.pkl","wb") as f:
    pickle.dump([model, scaler],f)

# For all val images
#predictions = get_and_write_predictions(scaler, model)
