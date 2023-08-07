import os
import numpy as np
import cv2

img_dir = '/Users/coenschoof/Desktop/BDD_img/bdd100k/images/100k/val'
lab_dir = '/Users/coenschoof/Desktop/BDD_labels/bdd100k/labels/det_20/det_val.json'
idx = 3
img_names = ['ca35c192-7f0eadba.jpg',
 'c08b49d7-164707bb.jpg',
 'c43eaa20-8450cd59.jpg',
 'c49d39a3-738d6240.jpg',
 'b25fd5d3-fa4bfca0.jpg']

img_path = os.path.join(img_dir, img_names[idx])
lab_path = os.path.join(lab_dir, img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')

print(lab_path)

# image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
label = np.loadtxt(lab_path, ndmin=2)
zeros = np.zeros((len(label),1)) + 0.00001
ones = np.ones((len(label),1))
label[:,[3,4]] = label[:,[3,4]] - 0.01
label[:, [3]] = np.minimum(ones, np.maximum(zeros, label[:, [3]]))
label[:, [4]] = np.minimum(ones, np.maximum(zeros, label[:, [4]]))

transformed = transform(image=image, bboxes=label[:, 1:], class_labels=label[:, 0])
image = transformed['image'].float()
bboxes = transformed['bboxes']
labels = transformed['class_labels']

merged_labels= np.array([np.concatenate(([np.array(labels[i])],np.array(bboxes[i]))) for i in range(0, len(labels))])
return image, merged_labels, img_names[idx]