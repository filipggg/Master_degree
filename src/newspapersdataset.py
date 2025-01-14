from PIL import Image
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tensor_transform
import sys
import pathlib
import warnings
sys.path.append("/".join(str(pathlib.Path(__file__).parent.resolve()).split('/')[:-2]))
from image_size import get_image_size  # source: https://github.com/scardine/image_size

# warnings
warnings.filterwarnings("ignore")


def prepare_data_for_dataloader(label_mapping, img_dir, in_list, expected_list=None, bbox_format='x0y0x1y1', scale=None, test=False):

    if label_mapping:
        last_label_ix = max(label_mapping.values())
    else:
        last_label_ix = 0

    df = pd.DataFrame()
    for i in range(len(in_list)):
        img_width, img_height = get_image_size.get_image_size(
            img_dir + in_list[i]
        )
        if isinstance(scale, list):
            new_img_width, new_img_height = scale[0], scale[1]
        elif isinstance(scale, int) or isinstance(scale, float):
            new_img_width, new_img_height = img_width * scale, img_height * scale
        else:
            new_img_width, new_img_height = img_width, img_height
        if test:
            temp_dict = {
                'file_name': int(in_list[i].split('.')[0]),
                'new_width': int(new_img_width),
                'new_height': int(new_img_height),
            }
            df = df.append(temp_dict, ignore_index=True)
        else:
            expected_list_split = expected_list[i].split(' ')
            for ii in range(len(expected_list_split)):
                file_num_name = in_list[i].split('.')[0]
                expected_list_split_2 = expected_list_split[ii].split(':')
                bbox = expected_list_split_2[1].split(',')

                original_label = expected_list_split_2[0]

                if original_label in label_mapping:
                    label = label_mapping[original_label]
                else:
                    last_label_ix += 1
                    label_mapping[original_label] = last_label_ix
                    label = last_label_ix

                x0, y0 = int(bbox[0]), int(bbox[1])
                x1, y1 = int(bbox[2]), int(bbox[3])
                if bbox_format == 'x0y0wh':
                    x1 += x0,
                    y1 += y0
                temp_dict = {
                    'file_name': int(file_num_name),
                    'class': int(label),
                    'x0': int(x0 / (img_width / new_img_width)),
                    'y0': int(y0 / (img_height / new_img_height)),
                    'x1': int(x1 / (img_width / new_img_width)),
                    'y1': int(y1 / (img_height / new_img_height)),
                    'new_width': int(new_img_width),
                    'new_height': int(new_img_height),
                }
                df = df.append(temp_dict, ignore_index=True)

    return df


def get_target(name, df, test=False):
    rows = df[df["file_name"] == int(name[:-4])]
    if test:
        return rows['file_name'].values, rows[['new_width', 'new_height']].values
    else:
        return rows['file_name'].values, rows["class"].values, rows[['x0', 'y0', 'x1', 'y1']].values, rows[['new_width', 'new_height']].values


class NewspapersDataset(Dataset):
  def __init__(self, images_path, df, scale=None, transforms=None, test=False):
    super(NewspapersDataset, self).__init__()
    self.df = df
    self.images_path = images_path
    self.scale = scale
    self.transforms = transforms
    self.test = test

  def __len__(self):
    return len(self.images_path)

  def __getitem__(self, idx):
    img_path = self.images_path[idx]
    img = Image.open(img_path)

    if not self.test:
        names, labels, boxes, new_image_size = get_target(img_path.split('/')[-1], self.df, test=False)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    else:
        names, new_image_size = get_target(img_path.split('/')[-1], self.df, test=True)

    if self.scale:
        img = img.resize((int(new_image_size[0, 0]), int(new_image_size[0, 1])))

    image_id = torch.tensor([idx])
    image_name = torch.as_tensor(names, dtype=torch.float32)
    new_image_size = torch.as_tensor(new_image_size, dtype=torch.float32)
    if not self.test:
        iscrowd = torch.zeros((boxes.shape[0],))
        area = torch.as_tensor(areas, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

    target = {}
    target['image_id'] = image_id
    target['image_name'] = image_name
    target['new_image_size'] = new_image_size
    if not self.test:
        target['labels'] = labels
        target['boxes'] = boxes
        target['area'] = area
        target['iscrowd'] = iscrowd

    if self.transforms:
        img = self.transforms(img)

    return img, target
