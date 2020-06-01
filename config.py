import os
import torch.cuda
import torchvision.transforms as transforms
import selector
import utils

config = {
    "num_workers": 8,
    "batch_size": 100,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "data_path": os.path.join(os.getcwd(), 'data'),
    "save_path": os.path.join(os.getcwd(), 'saved_results'),
}

method_config = {
    "TE": {
        "method": "TE",
        "process_list":
            ['train'],
        "temporal_ensemble": {
            'num_epochs': 300,
            'num_labels': 4000,
            'learning_rate_max':0.003,
            'rampup_length' : 80,
            'rampdown_length' : 50,
            'rampdown_beta1_target' : 0.5,
            'adam_beta1' : 0.9,
            'adam_beta2' : 0.999,
            'adam_epsilon' : 0.999,
            'ZCA' : True,
            'augment_translation' : 2,   #erase
            'augment_mirror' : True,  #erase
            'alpha' : 0.6,
            'std' : 0.15,
            'unsup_weight_max' : 30,
            'ramp_up_mult' : -5,
            'max_unlabeled_per_epoch' : None,
            'aux_tinyimg' : None
        },
        "package": "method.TE",
    }
}

data_config = {
    "mnist": {
        "dataset": "mnist",
        "total_classes": 10,
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "stl10": {
        "dataset": "stl10",
        "total_classes": 10,
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "cifar10": {
        "dataset": "cifar10",
        "total_classes": 10,
        "transform": {
            "train": transforms.Compose([
                transforms.RandomCrop(32, padding=2, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]),
            "test": transforms.Compose([
                transforms.ToTensor()
            ])
            #transforms.Normalize((0.4914, 0.4915, 0.4915), (0.1242, 0.1197, 0.1243))
        },
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "cifar100": {
        "dataset": "cifar100",
        "total_classes": 100,
        "classes": ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                    'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                    'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                    'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
                    'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
                    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    },
    "sub_image": {
        "dataset": "sub_image",
        "total_classes": 100,
        "classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                    '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
                    '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                    '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
                    '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56',
                    '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
                    '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78',
                    '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
                    '90', '91', '92', '93', '94', '95', '96', '97', '98', '99'],
    },
    "tiny_image": {
        "dataset": "tiny_image",
        "total_classes": 200,
        "classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                    '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
                    '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                    '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
                    '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56',
                    '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
                    '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78',
                    '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
                    '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100',
                    '101', '102', '103', '104', '105', '106', '107', '108', '109',
                    '110', '111', '112', '113', '114', '115', '116', '117', '118',
                    '119', '120', '121', '122', '123', '124', '125', '126', '127',
                    '128', '129', '130', '131', '132', '133', '134', '135', '136',
                    '137', '138', '139', '140', '141', '142', '143', '144', '145',
                    '146', '147', '148', '149', '150', '151', '152', '153', '154',
                    '155', '156', '157', '158', '159', '160', '161', '162', '163',
                    '164', '165', '166', '167', '168', '169', '170', '171', '172',
                    '173', '174', '175', '176', '177', '178', '179', '180', '181',
                    '182', '183', '184', '185', '186', '187', '188', '189', '190',
                    '191', '192', '193', '194', '195', '196', '197', '198', '199'],  # need to fix
    },
}
