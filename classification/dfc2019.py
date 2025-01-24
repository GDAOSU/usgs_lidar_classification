import numpy as np
from pathlib import Path
from os.path import join, exists, dirname, abspath
import logging
# import open3d as o3d
import pickle
from open3d.ml.datasets import Custom3D
from open3d.ml.utils import make_dir, Config, get_module, DATASET

from pathlib import Path

logger = logging.getLogger(__name__)


class DFC2019Split:
    def __init__(self, dataset, split='training'):
        # super().__init__(dataset, split=split)
        self.split=split
        self.dataset = dataset
        
        self.path_list = self.dataset.get_split_list(split)
        
        logger.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))
        
        if split in ['test']:
            sampler_cls = get_module('sampler', 'SemSegSpatiallyRegularSampler')
        else:
            sampler_cls = get_module('sampler', 'SemSegRandomSampler')
        self.sampler = sampler_cls(self)
        

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        data_path = self.path_list[idx]
        logger.debug("get_data called {}".format(data_path))

        #######################################################################
        # data preparation
        #######################################################################
        with open(data_path, 'rb') as f:
            pts, label = pickle.load(open(data_path, "rb"))[:2] # pts xyzIR: (N, 5), label: (N,)
        points = pts[:, :3].astype(np.float32)
        feat = pts[:, 3:].astype(np.float32)  # shape (N, 2)
        labels = label.astype(np.int32)
        
        # normalize the features
        # colum 0: intensity, colum 1: return number
        # intensity: by 90th percentile
        # return number: by 3
        feat_normalizer = np.array([np.median(feat[:,0]), 3.]).reshape(-1, 2).astype(np.float32)
        feat /= feat_normalizer
        
        data = {'point': points, 'feat': feat, 'label': labels}

        return data

    def get_attr(self, idx):  # TODO: check this function
        pc_path = Path(self.path_list[idx])
        name = pc_path.stem
        
        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr
    
class DFC2019(Custom3D.__base__):
    """DFC2019 dataset, used in visualizer, training, or test."""

    def __init__(self, dataset_path, name,
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 ignored_label_inds=[0],
                 test_folder=None,
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Semantic3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            num_points: The maximum number of points to use when splitting the dataset.
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         **kwargs)
        
        cfg = self.cfg
        
        self.dataset_path = Path(cfg.dataset_path)
        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array(cfg.ignored_label_inds)
        
        
        train_ids_path = self.dataset_path / 'train_ids.txt'
        val_ids_path = self.dataset_path / 'val_ids.txt'
        test_ids_path = self.dataset_path / 'test_ids.txt'
        
        # assert train_ids_path.exists(), f"train_ids.txt not found in {dataset_path}"
        # assert val_ids_path.exists(), f"val_ids.txt not found in {dataset_path}"
        # assert test_ids_path.exists(), f"test_ids.txt not found in {dataset_path}"
        
        # self.train_files = [ self.dataset_path / "trainval" / f"{i}.pkl" for i in train_ids_path.read_text().splitlines()]
        # self.val_files = [ self.dataset_path / "trainval" / f"{i}.pkl" for i in val_ids_path.read_text().splitlines()]
        # self.test_files = [ self.dataset_path / "test" / f"{i}.pkl" for i in test_ids_path.read_text().splitlines()]
        
        if test_folder is not None:
            self.test_files = list(sorted(Path(test_folder).glob("*.pkl")))

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'Unclassified',
            1: 'Ground',
            2: 'High_Vegetation',
            3: 'Building',
            4: 'Water',
            5: 'Bridge_Deck',
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return DFC2019Split(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

        return files

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        pred = np.array(pred)

        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        store_path = join(path, self.name, name + '.npy')
        make_dir(Path(store_path).parent)
        np.save(store_path, pred)
        logger.info("Saved {} in {}.".format(name, store_path))


DATASET._register_module(Custom3D)