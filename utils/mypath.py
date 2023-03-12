"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import yaml
from utils.common_config import get_val_transformations
from utils.CustomDataset import CustomDataset       

class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200'}
        assert(database in db_names)

        # if database == 'cifar-10':
        #     return 'Unsupervised-Classification/selflabel_cifar-10.pth.tar'

        if database == 'cifar-10':
            # Read config file
            with open('configs/selflabel/selflabel_cifar10.yml', 'r') as stream:
                config = yaml.safe_load(stream)
            config['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti

            # Get dataset
            transforms = get_val_transformations(config)
            dataset = CustomDataset('/home/fwojciak/projects/Unsupervised-Classification/costarica_2017_32x32', transforms)
        
            return dataset
        
        elif database == 'cifar-20':
            return '/path/to/cifar-20/'

        elif database == 'stl-10':
            return '/path/to/stl-10/'
        
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return 'Unsupervised-Classification/selflabel_cifar-10.pth.tar'
        
        else:
            raise NotImplementedError
