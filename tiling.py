from pathlib import Path
import pickle
import logging

import laspy
from tqdm import tqdm
import numpy as np

np.set_printoptions(precision=4, suppress=True)

logger = logging.getLogger(__name__)

USGS_CLASS_NAMES = {
    1: "Processed, but unclassified",
    2: "Bare earth",
    7: "Low noise",
    9: "Water",
    17: "Bridge deck",
    18: "High noise",
    20: "Ignored ground (typically breakline proximity)",
    21: "Snow (if present and identifiable)",
    22: "Temporal exclusion (typically nonfavored data in intertidal zones)"}

DFC2019_CLASS_NAMES = {
    0: "Unclassified",
    2: "Ground",
    5: "High Vegetation",
    6: "Building",
    9: "Water",
    17: "Bridge Deck"}

COMPACT_IDX = {
    0: 0,
    2: 1,
    5: 2,
    6: 3,
    9: 4,
    17: 5}
COMPACT_IDX_2_ORIGIN_IDX = {v: k for k, v in COMPACT_IDX.items()}

USGS_2_DFC2019 = {i: 0 for i in range(256)}
USGS_2_DFC2019 .update({2: 2, 9: 9, 17: 17})


def _get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Tiling Point Cloud')
    parser.add_argument('--block_width', type=float, default=512, help='Block Width (meters)')
    parser.add_argument('--in_path', type=str, required=True, help='Input Point Cloud Path')
    parser.add_argument('--out_dir', type=str, help='Output Directory')
    return parser


def tiling(in_path, out_dir, block_width):
    logging.basicConfig(level=logging.INFO)

    in_path = Path(in_path)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f">>>>>>>>>>>>>>>>>>")
    logger.info(f"Input: {in_path}")
    logger.info(f"Output: {out_dir}")

    if in_path.suffix in (".laz", ".las"):
        las = laspy.read(in_path)
        scales = las.header.scale
        offsets = las.header.offset

        xyz_min = np.array(las.header.min)
        xyz_max = np.array(las.header.max)
        xyz_mean = 0.5*(xyz_min + xyz_max)

        coord = np.stack([las.X, las.Y, las.Z], axis=1).astype(np.float64) * scales + offsets
        feat = np.stack([las.intensity, las.return_num], axis=1)
        label = np.array(las.classification, dtype=np.uint8)

        label = np.vectorize(USGS_2_DFC2019.get)(label)
        label = np.vectorize(COMPACT_IDX.get)(label)

    elif in_path.suffix == ".pkl":
        data, label, xyz_mean = pickle.load(open(in_path, "rb"))
        coord = data[:, :3]
        feat = data[:, 3:]
        xyz_min = np.min(coord, axis=0)
        xyz_max = np.max(coord, axis=0)
    else:
        raise NotImplementedError(f"Unknown file type: {in_path.suffix}")

    xyz_range = np.abs(xyz_max - xyz_min)

    tile_x_count = max(1, int(np.round(xyz_range[0] / block_width)))
    tile_y_count = max(1, int(np.round(xyz_range[1] / block_width)))

    logger.info(f">> Range: {xyz_range} Offset: {xyz_mean}")
    logger.info(f">> Tiles: {tile_x_count} x {tile_y_count}")

    tile_x = np.linspace(xyz_min[0], xyz_max[0], tile_x_count+1)
    tile_y = np.linspace(xyz_min[1], xyz_max[1], tile_y_count+1)

    logger.info(">> Intensity Distribution:")
    logger.info(f"  0%: {np.percentile(feat[:,0], 0)}")
    logger.info(f"  50%: {np.percentile(feat[:,0], 50)}")
    logger.info(f"  90%: {np.percentile(feat[:,0], 90)}")
    logger.info(f"  95%: {np.percentile(feat[:,0], 95)}")
    logger.info(f"  100%: {np.percentile(feat[:,0], 100)}")
    logger.info(">> Return Number Distribution:")
    logger.info(f"  Min: {np.min(feat[:,1])} Max: {np.max(feat[:,1])}")
    if label is not None:
        logger.info(">> Label Distribution:")
        logger.info(dict(zip(*np.unique(label, return_counts=True))))

    # Create tiles
    for xi, yi in tqdm(np.ndindex(tile_x_count, tile_y_count), total=tile_x_count*tile_y_count, desc=f"Tiling {in_path.stem}"):
        out_tile_path = out_dir / f"x{xi}_y{yi}.pkl"
        if out_tile_path.exists():
            continue
        tile_x_min = tile_x[xi]
        tile_x_max = tile_x[xi+1]
        tile_y_min = tile_y[yi]
        tile_y_max = tile_y[yi+1]

        tile_idx = np.where((coord[:, 0] >= tile_x_min) & (coord[:, 0] < tile_x_max) & (coord[:, 1] >= tile_y_min) & (coord[:, 1] < tile_y_max))[0]

        tile_coord = (coord[tile_idx] - xyz_mean).astype(np.float32)
        tile_feat = feat[tile_idx].astype(np.float32)
        tile_label = None if label is None else label[tile_idx].astype(np.uint8)

        tile_pts = np.hstack([tile_coord, tile_feat])

        pickle.dump((tile_pts.astype(np.float32),
                    tile_label,
                    xyz_mean.astype(np.float64),
                    tile_idx.astype(np.uint32)),
                    open(str(out_tile_path), "wb"))
    logger.info(f"<<<<<<<<<<<<<<<<<<<<")
    return 0


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    tiling(args.in_path, args.out_dir, args.block_width)
