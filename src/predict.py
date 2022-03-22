#!/usr/bin/env python3

import sys
import torch
from functions import from_tsv_to_list
from newspapersdataset import NewspapersDataset
from newspapersdataset import prepare_data_for_dataloader
from parameters import parameters
import torchvision.transforms as T
from torch.utils.data import DataLoader
from functions import collate_fn
from test_model import model_predict
import pandas as pd
import numpy as np
import ast
import get_image_size
from functions import save_list_to_tsv_file

in_file = sys.argv[1]
out_file = sys.argv[2]

data_transform = T.Compose([
    T.Grayscale(num_output_channels=parameters['channel']),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
    ])

model = torch.load(parameters['main_dir']+'saved_models/model.pth')
in_test = from_tsv_to_list(in_file)
test_paths = [parameters['image_dir']+path for path in in_test]
data_test = prepare_data_for_dataloader(
    img_dir=parameters['image_dir'],
    in_list=in_test,
    scale=parameters['rescale'],
    test=True,
)
dataset_test = NewspapersDataset(
    df=data_test,
    images_path=test_paths,
    scale=parameters['rescale'],
    transforms=data_transform,
    test=True,
)
test_dataloader = DataLoader(
    dataset_test,
    batch_size=parameters['batch_size'],
    shuffle=parameters['shuffle'],
    collate_fn=collate_fn,
    num_workers=parameters['num_workers'],
)

csv_file = parameters['main_dir']+'model_output/test_model_output.csv'

model_predict(
    model=model,
    test_dataloader=test_dataloader,
    gpu=parameters['gpu'],
    save_path=csv_file,
)


test_df = pd.read_csv(csv_file, index_col=0)
in_test = from_tsv_to_list(in_file)


def parse_model_outcome(model_outcome_df, in_file, image_directory):
    # getting images size before rescale
    img_old_sizes_list = []
    for i in range(len(in_file)):
        img_width, img_height = get_image_size.get_image_size(
            image_directory + in_file[i]
        )
        img_old_sizes_list.append([img_width, img_height])

    model_outcome_df['old_image_size'] = img_old_sizes_list

    scaler_width, scaler_height = [], []
    for i in range(len(model_outcome_df)):
        old_image_size_width = model_outcome_df['old_image_size'][i][0]
        old_image_size_height = model_outcome_df['old_image_size'][i][1]
        new_image_size_width = np.float(ast.literal_eval(model_outcome_df['new_image_size'][i])[0][0])
        new_image_size_height = np.float(ast.literal_eval(model_outcome_df['new_image_size'][i])[0][0])

        scaler_width.append(np.float(old_image_size_width)/np.float(new_image_size_width))
        scaler_height.append(np.float(old_image_size_height)/np.float(new_image_size_height))

    out_list = []
    for i in range(len(model_outcome_df)):
        pred_labels = ast.literal_eval(model_outcome_df['predicted_labels'][i])
        pred_boxes = ast.literal_eval(model_outcome_df['predicted_boxes'][i])
        out_str = ''
        for ii in range(len(pred_labels)):
            if int(pred_labels[ii]) == 1:
                label = 'photograph'
            elif int(pred_labels[ii]) == 2:
                label = 'illustration'
            elif int(pred_labels[ii]) == 3:
                label = 'map'
            elif int(pred_labels[ii]) == 4:
                label = 'cartoon'
            elif int(pred_labels[ii]) == 5:
                label = 'editorial_cartoon'
            elif int(pred_labels[ii]) == 6:
                label = 'headline'
            elif int(pred_labels[ii]) == 7:
                label = 'advertisement'
            x0 = str(int(round(pred_boxes[ii][0], 0)*scaler_width[i]))
            y0 = str(int(round(pred_boxes[ii][1], 0)*scaler_height[i]))
            x1 = str(int(round(pred_boxes[ii][2], 0)*scaler_width[i]))
            y1 = str(int(round(pred_boxes[ii][3], 0)*scaler_height[i]))

            out_str = out_str + f'{label}:{x0},{y0},{x1},{y1} '

        out_str = out_str.strip(" ")
        out_list.append(out_str)

    return out_list


out_list_test = parse_model_outcome(test_df, in_test, parameters['image_dir'])
save_list_to_tsv_file(out_file, out_list_test)
