import laspy
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import pickle 
from glob import glob

def write_las(xyz, intensity, return_num, classification, xyz_offset, out_las_path):
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = xyz_offset
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)

    las.x = xyz[:,0]
    las.y = xyz[:,1]
    las.z = xyz[:,2]
    las.intensity = intensity
    return_num = return_num.astype(np.uint8)
    las.return_num = return_num
    las.classification = classification

    las.write(out_las_path.open('wb'))

def _get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_pkl_dir", type=Path, required=True)
    parser.add_argument("--in_label_dir", type=Path, required=True)
    parser.add_argument("--in_label_suffix", type=str, default='pred')
    parser.add_argument("--out_las_path", type=Path, required=True)
    return parser
  
def stitching():
    parser = _get_parser()
    args = parser.parse_args()
    
    in_pkl_dir = args.in_pkl_dir
    in_label_dir = args.in_label_dir
    in_label_suffix = args.in_label_suffix
    out_las_path = args.out_las_path
    
    # in_pkl_dir = Path("/local/scratch0/hanyang/Codes/point_cloud_visualizer/tmp_dir/tiles")
    in_pkl_paths = list(in_pkl_dir.glob("x*_y*.pkl"))
    
    in_label_dir = Path("tmp_dir/tile_classification_results")
    # in_label_suffix = "label"
    # out_las_path = in_pkl_dir / f"{in_label_suffix}.las"

    data_arr = []
    classification_arr = []
    for in_pkl_path in tqdm(in_pkl_paths,desc="Reading pkl"):
      data, label, xyz_offset = pickle.load(open(in_pkl_path, "rb"))[:3]
      data = data.astype(np.float64)
      if len(data) == 0:
        continue
      data[:,:3] += xyz_offset
      data_arr.append(data)
      
      if in_label_suffix is None:
        classification_arr.append(label)
      else:
        in_label_path = sorted(list(in_label_dir.glob(f"{Path(in_pkl_path).stem}*_{in_label_suffix}.npy")))[0]
        classification = np.load(in_label_path)
        classification_arr.append(classification)


    data = np.concatenate(data_arr, axis=0)
    classification = np.concatenate(classification_arr, axis=0)

    xyz = data[:,:3].astype(np.float64)
    intensity = data[:,3]
    return_num = data[:,4]
    
    write_las(xyz, intensity, return_num, classification, xyz_offset, out_las_path)
    print("Done")

if __name__ == "__main__":
    stitching()