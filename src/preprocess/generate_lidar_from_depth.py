import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitti_util
import numpy as np
from PIL import Image


def project_disp_to_depth(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)
    depth = depth_png.astype(np.float) / 256.
    return depth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Lidar')
    parser.add_argument('--calib_dir', type=str,
                        default='../../data/dataset/kitti_raw/2011_09_30/2011_09_30_calib')
    parser.add_argument('--depth_dir', type=str,
                        default='../../data/dataset/kitti_depth/kitti_09_30_20_prediction_validpix_02')
    parser.add_argument('--save_dir', type=str,
                        default='../../data/dataset/kitti_raw/2011_09_30/2011_09_30_drive_0020_sync/velodyne_points/data')
    parser.add_argument('--max_high', type=int, default=1)
    args = parser.parse_args()

    assert os.path.isdir(args.depth_dir)
    assert os.path.isdir(args.calib_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    depths = [x for x in os.listdir(args.depth_dir) if x[-3:] == 'png' and 'std' not in x]
    depths = sorted(depths)
    calib = kitti_util.Calibration(args.calib_dir)

    for fn in depths:
        predix = fn[:-4]
        print(predix)
        # calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
        # calib = kitti_util.Calibration(args.calib_dir)
        # depth_map = np.load(args.depth_dir + '/' + fn,encoding='bytes',allow_pickle=True)
        depth_map=depth_read(args.depth_dir + '/' + fn)
        lidar = project_disp_to_depth(calib, depth_map, args.max_high)
        # pad 1 in the indensity dimension
        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
        lidar = lidar.astype(np.float32)
        lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))
        print('Finish Depth {}'.format(predix))

