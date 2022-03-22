#!/usr/bin/env python3

from parameters import parameters
import yaml

gonito_metadata = {
    'description': 'Training only classifier of pretrained resnet-50',
    'tags': ['neural-network',
             'cnn',
             'faster-r-cnn'],
    'params': parameters
}

with open('gonito.yaml', 'w') as file:
    yaml.dump(gonito_metadata, file)
