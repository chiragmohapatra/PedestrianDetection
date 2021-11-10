# File paths
import torch
import torchvision
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os

import argparse

parser = argparse.ArgumentParser(description='Evaluation script')
parser.add_argument('--root', type=str, help='path to root dir')
parser.add_argument('--test', type=str, help='path to test json')
parser.add_argument('--out', type=str, help='path to out json')

args = parser.parse_args()

#train_json_path = 'PennFudanPed_train.json'
#test_json_path = 'PennFudanPed_val.json'
root_dir = args.root
test_json_path = args.test
#output_path = 'preds.json'
output_path = args.out

# rcnn dataloader
class PedDataset(Dataset):
  def __init__(self,images_data):
    self.img_paths = []
    self.ids = []

    for img_data in images_data:
      self.img_paths.append(os.path.join(root_dir, img_data['file_name']))
      self.ids.append((img_data['id']))

  def __len__(self):
    return len(self.ids)

  def __getitem__(self,index):
    img_path = self.img_paths[index]
    img_id = self.ids[index]

    img = Image.open(img_path)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)

    return img , img_id


# faster rcnn part 3

f = open(test_json_path, 'r')
test_data = json.load(f)
f.close()

test_loader = DataLoader(PedDataset(test_data['images']) , batch_size=1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
threshold = 0.5
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) 
#model = model.cuda()
model = model.to(device)
model.eval()

predictions = []

for i,(img, img_id) in enumerate(test_loader):
  #img = img.cuda()
  img = img.to(device)
  pred = model(img)
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
  pred_score = list(pred[0]['scores'].cpu().detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  pred_score = pred_score[:pred_t+1]

  for i in range(len(pred_boxes)):
    if pred_class[i] == 'person':
        (x1, y1), (x2, y2) = pred_boxes[i]
        bbox = list(map(float, [x1, y1, x2-x1, y2-y1]))
        predictions.append({
                    'image_id':img_id.item(),
                    'category_id':1,
                    'bbox':bbox,
                    'score':float(pred_score[i])
                })
        
with open(output_path, 'w') as f_out:
    json.dump(predictions , f_out)
