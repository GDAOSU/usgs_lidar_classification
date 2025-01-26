import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from point_transformer.util import config
from point_transformer.util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from point_transformer.util.voxelize import voxelize

random.seed(123)
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        default='point_transformer/dfc2019_pointtransformer_repro.yaml',
                        help='config file')
    parser.add_argument('--data_path', type=str, default=None, help='path to dataset *.pkl')
    parser.add_argument('--data_dir', type=str, default='tmp_dir/tiles', help='dataset name')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--save_folder', help='result folder',
                        default='tmp_dir/tile_classification_results', type=str)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg['inference_data_path'] = args.data_path
    cfg['inference_data_dir'] = args.data_dir
    cfg['save_folder'] = args.save_folder
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.arch == 'pointtransformer_seg_repro':
        from point_transformer.model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))

    if args.inference_data_path is None and args.inference_data_dir is None:
        raise Exception("data_path and data_dir cannot be both None")

    model = Model(c=args.fea_dim, k=args.classes).cuda()
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    inference(model, criterion, names)


def data_prepare():
    if args.data_name == 's3dis':
        data_list = sorted(os.listdir(args.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    elif args.data_name in ['dfc2019', 'us3d']:
        if args.inference_data_path is not None:
            args.data_root = Path(args.inference_data_path).parent
            data_list = [Path(args.inference_data_path).stem]
        else:
            args.data_root = Path(args.inference_data_dir)
            data_list = sorted([i.stem for i in args.data_root.glob("*.pkl")])
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    print("Totally {} samples in val set.".format(len(data_list)))
    return data_list


def data_load(data_name):
    if args.data_name == 's3dis':
        data_path = os.path.join(args.data_root, data_name + '.npy')
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
    elif args.data_name in ['dfc2019', 'us3d']:
        dataset_dir = Path(args.data_root)
        data_path = dataset_dir / f"{data_name}.pkl"
        pts, label = pickle.load(open(data_path, "rb"))[:2]  # pts xyzIR: (N, 5), label: (N,)

        coord, feat, label = pts[:, 0:3], pts[:, 3:5], np.array(label).reshape(-1)

        intensity_divisor = np.nanpercentile(feat[:, 0], 90)
        return_divisor = 3.
        feat /= np.array([intensity_divisor, return_divisor]).reshape(-1, 2).astype(np.float32)

    idx_data = []
    if args.voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:
        idx_data.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_data


def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    if args.data_name == 's3dis':
        feat = feat / 255.
    return coord, feat


def inference(model, criterion, names):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    args.voxel_max = args.test_voxel_max
    model.eval()
    check_makedirs(args.save_folder)
    pred_save, label_save = [], []

    data_list = data_prepare()
    for idx, item in enumerate(data_list):
        end = time.time()
        pred_save_path = os.path.join(args.save_folder, '{}_{}_pred.npy'.format(item, args.epoch))
        label_save_path = os.path.join(args.save_folder, '{}_{}_label.npy'.format(item, args.epoch))
        if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
            logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(data_list), item))
            pred, label = np.load(pred_save_path), np.load(label_save_path)
        else:
            try:
                coord, feat, label, idx_data = data_load(item)
            except Exception as e:
                logger.error(f"{idx+1}/{len(data_list)}: {item} failed to load data, {e}")
                continue

            pred = torch.zeros((label.size, args.classes)).cuda()
            idx_size = len(idx_data)
            idx_list, coord_list, feat_list, offset_list = [], [], [], []
            for i in range(idx_size):
                logger.info('{}/{}: {}/{}/{}, {}'.format(idx + 1, len(data_list), i + 1, idx_size, idx_data[0].shape[0], item))
                idx_part = idx_data[i]
                coord_part, feat_part = coord[idx_part], feat[idx_part]
                if args.voxel_max and coord_part.shape[0] > args.voxel_max:
                    coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
                    while idx_uni.size != idx_part.shape[0]:
                        init_idx = np.argmin(coord_p)
                        dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                        idx_crop = np.argsort(dist)[:args.voxel_max]
                        coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
                        dist = dist[idx_crop]
                        delta = np.square(1 - dist / np.max(dist))
                        coord_p[idx_crop] += delta
                        coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
                        idx_list.append(idx_sub), coord_list.append(coord_sub), feat_list.append(feat_sub), offset_list.append(idx_sub.size)
                        idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
                else:
                    coord_part, feat_part = input_normalize(coord_part, feat_part)
                    idx_list.append(idx_part), coord_list.append(coord_part), feat_list.append(feat_part), offset_list.append(idx_part.size)
            batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
                idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], feat_list[s_i:e_i], offset_list[s_i:e_i]
                idx_part = np.concatenate(idx_part)
                coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
                feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
                offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = model([coord_part, feat_part, offset_part])  # (n, k)
                torch.cuda.empty_cache()
                idx_part = torch.LongTensor(idx_part).cuda(non_blocking=True)
                pred[idx_part, :] += pred_part
                logger.info('Test: {}/{}, {}/{}, {}/{}'.format(idx + 1, len(data_list), e_i, len(idx_list), args.voxel_max, idx_part.shape[0]))

            if label is None:
                loss = np.nan
            else:
                loss = criterion(pred, torch.LongTensor(label).cuda(non_blocking=True))  # for reference
            pred = pred.max(1)[1].data.cpu().numpy()

        if label is None:
            accuracy = np.nan
        else:
            # calculation 1: add per room predictions
            intersection, union, _, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            accuracy = sum(intersection) / (sum(target) + 1e-10)

        batch_time.update(time.time() - end)
        logger.info('Test: [{}/{}]-{} '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Accuracy {accuracy:.4f}.'.format(idx + 1, len(data_list), pred.size, batch_time=batch_time, accuracy=accuracy))
        pred_save.append(pred)
        label_save.append(label)
        np.save(pred_save_path, pred)
        np.save(label_save_path, label)

    with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
        pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
        pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if label is None:
        # calculation 1
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU1 = np.mean(iou_class)
        mAcc1 = np.mean(accuracy_class)
        allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        # calculation 2
        intersection, union, _, target = intersectionAndUnion(np.concatenate(pred_save), np.concatenate(label_save), args.classes, args.ignore_label)
        iou_class = intersection / (union + 1e-10)
        accuracy_class = intersection / (target + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection) / (sum(target) + 1e-10)
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
