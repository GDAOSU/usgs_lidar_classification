import warnings
warnings.filterwarnings('ignore')
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

from open3d.ml.torch.pipelines import SemanticSegmentation

from open3d.ml.torch.models import RandLANet, PointTransformer

from classification.dfc2019 import DFC2019

from pathlib import Path
import sys
import pickle
import numpy as np
from tqdm import trange
import argparse
import time
import os


cfg_file = os.path.join(os.path.dirname(__file__), 'classification', 'randlanet_us3d.yml')
cfg_name = 'RandLANet_US3D_torch'
ckpt_path = f"logs/{cfg_name}/checkpoint/ckpt_00255.pth"
ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)

###########
def inference(test_folder, test_result_folder):
    
    test_folder = Path(test_folder)
    test_result_folder = Path(test_result_folder)
    
    _start = time.time()
    cfg = _ml3d.utils.Config.load_from_file(str(cfg_file))

    if cfg.model.name == "PointTransformer":
        model = PointTransformer(**cfg.model, framework='torch')
    elif cfg.model.name == "RandLANet":
        model = RandLANet(**cfg.model, framework='torch')
    else:
        raise ValueError(f"Model {cfg.model.name} not supported")

    pipeline = ml3d.pipelines.SemanticSegmentation(model=model, **cfg.pipeline)
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    test_result_folder = Path(test_result_folder)
    test_result_folder.mkdir(parents=True, exist_ok=True)

    dataset = DFC2019(**cfg.dataset, framework='torch',
                    test_folder = test_folder)

    pred_save, label_save = [], []

    test_split = dataset.get_split('test')
    for i in trange(len(test_split)):
        data = test_split.get_data(i)
        pred_save_path = test_result_folder / f"{test_split.get_attr(i)['name']}_pred.npy"
        label_save_path = test_result_folder / f"{test_split.get_attr(i)['name']}_label.npy"
        

        label = data['label']
        
        result = pipeline.run_inference(data)
        
        pred= result['predict_labels'] + 1
        
        pred_save.append(pred); label_save.append(label)
        np.save(pred_save_path, pred); np.save(label_save_path, label)

            
    total_pred_save_path = test_result_folder / "pred.pickle"
    total_label_save_path = test_result_folder / "label.pickle"

    pickle.dump({'pred': pred_save}, open(total_pred_save_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump({'label': label_save}, open(total_label_save_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    _duration = time.time() - _start
    print(f"Total time: {_duration:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_folder", type=str, help="Path to the test folder")
    parser.add_argument("--test_result_folder", type=str, help="Path to the test result folder")
    args = parser.parse_args()
    inference(args.test_folder, args.test_result_folder)