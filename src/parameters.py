
# hyperparameters and more
parameters = {
    'channel': 1, # 3 <= RGB, 1 <= greyscale
    'num_classes': 8, # 7 classes, but there is also one for background
    'learning_rate': 3e-4,
    'batch_size': 2,
    'num_epochs': 1,
    'rescale': [1000, 1000], # if float, each image will be multiplied by it, if list [width, height] each image will be scaled to that size (concerns both images + annotations)
    'shuffle': True,
    'weight_decay': 0, # regularization
    'lr_scheduler': True, # lr scheduler
    'lr_step_size': 5, # lr scheduler step
    'lr_gamma': .4, # lr step multiplier
    'trainable_backbone_layers': 5, # 5 <= all, 0 <= any
    'num_workers': 2,
    'main_dir': './models/',
    'image_dir': './images/',
    'annotations_dir': './',
    'train': True,
    'test': True,
    'val': True,
    'gpu': True,
}
