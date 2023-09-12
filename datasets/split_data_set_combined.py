import cv2
import fnmatch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from torch.utils.data import Dataset
import random


class SplitDatasetCombined_BDD:
    def __init__(self, img_dir, lab_dir, max_lab, img_size, transform=None, collate_fn=None):
        self.dataset_train = CustomDataset(type_ds='train',
                                     img_dir=img_dir,
                                     lab_dir=lab_dir,
                                     max_lab=max_lab,
                                     img_size=img_size,
                                     transform=transform)
        self.dataset_val = CustomDataset(type_ds='val',
                                           img_dir=img_dir,
                                           lab_dir=lab_dir,
                                           max_lab=max_lab,
                                           img_size=img_size,
                                           transform=transform)
        self.dataset_test = CustomDataset(type_ds='test',
                                           img_dir=img_dir,
                                           lab_dir=lab_dir,
                                           max_lab=max_lab,
                                           img_size=img_size,
                                           transform=transform)
        self.img_dir = img_dir

        self.collate_fn = collate_fn

    #maakt de dataloaders aan voor train, val en test
    def __call__(self, val_split, shuffle_dataset, random_seed, batch_size, *args, **kwargs):

        #returnt de indices voor train,val en test
        #[0:1350], [1350:1500], [1500:2000] respectively
        train_indices, val_indices, test_indices = self.create_random_indices(val_split)
        # print(train_indices)
        # print(val_indices)
        # print(test_indices)

        #waarom zou je nog een keer shufflen? dat is al in create_random_indices gedaan
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        #self.dataset_train is nu nog een tuple van (tensor, array, string), waarbij tensor = image, array = labels en string = image path
        #self.data_train[0] bevat nu 1 datapunt
        #batch_size = 8
        #sampler = gives random permutation of 8 train_indices each time train_indices is called
        train_loader = DataLoader(self.dataset_train, batch_size=batch_size,  sampler=train_sampler, collate_fn=self.collate_fn)
        validation_loader = DataLoader(self.dataset_val, batch_size=batch_size, sampler=valid_sampler, collate_fn=self.collate_fn)
        test_loader = DataLoader(self.dataset_test, batch_size=1, sampler=test_sampler, collate_fn=self.collate_fn)

        return train_loader, validation_loader, test_loader

    def create_random_indices(self,val_split):
        # To evaluate our adversarial perturbation, we randomly
        # chose 2,000 images from the validation set of each of the
        # datasets (BDD, MTSD, and PASCAL). For each dataset, we
        # used 1,500 images to train the UAP and then examined its
        # effectiveness on the remaining 500 images

        #De UAPs worden gemaakt obv de validation set, which is 10k long
        #maak een lijst van [0,1,2,3,...,9999]
        all_indices = [i for i in range(10000)]
        total = 2000
        #neemt 2000 random indices uit all_indices
        data_set_indices = random.choices(all_indices, k=total)
        train_val = 1500
        #split_index = 1500 * 0.9 = 1350
        split_index = int(train_val * (1-val_split))
        #datapunt 0 tot 1350
        train_indices = data_set_indices[0:split_index]
        #datapunt 1350 tot 1500 (150 lang)
        val_indices = data_set_indices[split_index:train_val]
        #datapunt 1500 tot 2000 (500 lang)
        test_indices = data_set_indices[train_val:total]

        return train_indices, val_indices, test_indices

class CustomDataset(Dataset):
    def __init__(self, type_ds, img_dir, lab_dir, max_lab, img_size, shuffle=True, transform=None):
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.img_size = img_size
        self.shuffle = shuffle
        self.type_ds = type_ds
        self.img_names = self.get_image_names()
        self.img_paths = self.get_image_paths()
        self.lab_paths = self.get_lab_paths()
        self.max_n_labels = max_lab
        self.transform = transform


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        print(idx)
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        #numpyarray (720, 1280, 3) (y-as, x-as, RGB)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        #voorbeeld van label .txt file
        # 0 0.491778 0.432859 0.009186 0.009526
        # 2 0.274377 0.504889 0.039806 0.057157
        # 2 0.227299 0.502167 0.071191 0.078931
        # 2 0.168355 0.518498 0.081908 0.087097
        label = np.loadtxt(lab_path, ndmin=2)
        zeros = np.zeros((len(label),1)) + 0.00001
        ones = np.ones((len(label),1))
        #Trek 0.01 af van de laatste kolommen
        #WAAROM?
        label[:,[3,4]] = label[:,[3,4]] - 0.01
        label[:, [3]] = np.minimum(ones, np.maximum(zeros, label[:, [3]]))
        label[:, [4]] = np.minimum(ones, np.maximum(zeros, label[:, [4]]))

        #transformeer de data zodat het 640x640 is
        transformed = self.transform(image=image, bboxes=label[:, 1:], class_labels=label[:, 0])

        image = transformed['image'].float()
        bboxes = transformed['bboxes']
        labels = transformed['class_labels']

        merged_labels= np.array([np.concatenate(([np.array(labels[i])],np.array(bboxes[i]))) for i in range(0, len(labels))])
        return image, merged_labels, self.img_names[idx]



    def get_image_names(self):
        png_images = fnmatch.filter(os.listdir(self.img_dir), '*.png')
        jpg_images = fnmatch.filter(os.listdir(self.img_dir), '*.jpg')
        n_png_images = len(png_images)
        n_jpg_images = len(jpg_images)

        return png_images + jpg_images


    def get_image_paths(self):
        img_paths = []
        for img_name in self.img_names:
            img_paths.append(os.path.join(self.img_dir, img_name))
        return img_paths

    def get_lab_paths(self):
        lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            lab_paths.append(lab_path)
        return lab_paths

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if pad_size > 0:
            padded_lab = F.pad(lab, [0, 0, 0, pad_size], value=-1)
        else:
            padded_lab = lab
        return padded_lab




