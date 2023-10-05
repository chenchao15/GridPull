# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:44:22 2020

@author: Administrator
"""

import numpy as np

import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

import os 
import shutil
import random
import math
import scipy.io as sio
import time
from skimage import measure
# import binvox_rw
import argparse
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.utils.libkdtree import KDTree
from sample_func import get_sample, init_sphere, init_smooth_grid_index, bigger
import re
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--train',action='store_true', default=False)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--class_idx', type=str, default="026911156")
parser.add_argument('--save_idx', type=int, default=-1)
parser.add_argument('--CUDA', type=int, default=0)
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--dataset', type=str, default="shapenet")
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.7)
parser.add_argument('--base_lr', type=float, default=0.1)
parser.add_argument('--vox_loss_weight', type=float, default=10)
parser.add_argument('--grad_loss_weight', type=float, default=0.1)
parser.add_argument('--sdf_loss_weight', type=float, default=1.0)
parser.add_argument('--sphere_radius', type=float, default=0.5)
parser.add_argument('--level', type=int, default=15)
parser.add_argument('--obj_ind', type=int, default=0)
a = parser.parse_args()


cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx


BS = 1
POINT_NUM = 50000
POINT_NUM_GT = 50000
INPUT_DIR = a.data_dir
INDEX = a.index
GRID_SIZE = 1621
VOX_SIZE =256
SCALE = a.scale
LR = a.lr
BASE_LR = a.base_lr
VOX_LOSS_WEIGHT = a.vox_loss_weight
GRAD_LOSS_WEIGHT = a.grad_loss_weight
SDF_LOSS_WEIGHT = a.sdf_loss_weight
SAMPLE_TYPE= 10 # a.sample_type
TEST_VOX_SIZE = 256
OUTPUT_DIR = a.out_dir

listdir = os.path.join(INPUT_DIR, 'list.txt')
with open(listdir, 'r') as f:
    obj_names = f.readlines()
    objn = obj_names[a.obj_ind].strip('\n')
    objn = objn.split('.')[0]
obj_name = os.path.join(INPUT_DIR, 'famous_noisefree', '04_pts', objn+'.xyz.npy')


if obj_name[-4:] == '.xyz':
    gttxt = np.loadtxt(obj_name)
    gttxt = gttxt[:, :3]
elif 'txt' in obj_name:
    gttxt = np.loadtxt(obj_name)
elif 'ply' in obj_name:
    gttxt = trimesh.load(obj_name)
    gttxt = gttxt.vertices
elif 'npy' in obj_name:
    gttxt = np.load(obj_name)
    gttxt = np.float64(gttxt)
origin_gt = gttxt
minn = np.min(gttxt)
maxn = np.max(gttxt)
gttxt = (gttxt - minn) / (maxn - minn)
gttxt -= 0.5
gttxt_bigger = bigger(gttxt, VOX_SIZE)

sphere_init_sdf = init_sphere(radius=a.sphere_radius, size=VOX_SIZE) # np.load('vox_np_256.npy')
useful_index, useful_index_test, useful_index_weight = init_smooth_grid_index(pc=gttxt_bigger, size=VOX_SIZE, level=a.level) # np.loadtxt('usefull_index_gt_armadillo.xyz')

kdtree = KDTree(gttxt)

if(a.dataset=="shapenet" or a.dataset=='other'):
    GT_DIR = './origin_data/' + a.class_idx + '/'
if(a.dataset=="famous"):
    GT_DIR = './data/famous_noisefree/03_meshes/'
if(a.dataset=="ABC"):
    GT_DIR = './data/abc_noisefree/03_meshes/'

GT_DIR = os.path.join(INPUT_DIR, 'famous_dense', '03_meshes/')

TRAIN = a.train
bd = 1.0 
test_bd = 0.55

if(TRAIN):
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print ('test_res_dir: deleted and then created!')
    os.makedirs(OUTPUT_DIR)
else:
    POINT_NUM =TEST_VOX_SIZE * TEST_VOX_SIZE

np.savetxt(os.path.join(OUTPUT_DIR, 'gt.txt'), origin_gt)


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        rmals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

#        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
#        # Handle normals that point into wrong direction gracefully
#        # (mostly due to mehtod not caring about this in generation)
#        normals_dot_product = np.abs(normals_dot_product)
        
        normals_dot_product = np.abs(normals_tgt[idx] * normals_src)
        normals_dot_product = normals_dot_product.sum(axis=-1)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty


        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()
        #print(completeness,accuracy,completeness2,accuracy2)
        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        print('chamferL2:',chamferL2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)
        print('normals_correctness:',normals_correctness,'chamferL1:',chamferL1)
        return normals_correctness, chamferL1, chamferL2

def safe_norm_np(x, epsilon=1e-12, axis=1):
    return np.sqrt(np.sum(x*x, axis=axis) + epsilon)

def safe_norm(x, epsilon=1e-12, axis=None):
  return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)

def boundingbox(x,y,z):
    return min(x),max(x),min(y),max(y),min(z),max(z)

    

def chamfer_distance_tf_None(array1, array2):
    array1 = tf.reshape(array1,[-1,3])
    array2 = tf.reshape(array2,[-1,3])
    av_dist1 = av_dist_None(array1, array2)
    av_dist2 = av_dist_None(array2, array1)
    return av_dist1+av_dist2

def distance_matrix_None(array1, array2, num_point, num_features = 3):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances

def av_dist_None(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix_None(array1, array2,points_input_num[0,0])
    distances = tf.reduce_min(distances, axis=1)
    distances = tf.reduce_mean(distances)
    return distances


def get_reg_loss():
    res = tf.zeros([])
    vars = tf.trainable_variables()
    for v in vars:
        res += tf.nn.l2_loss(v)
    return res


def chamfer_distance_tf_None2(p, q):
    from nn_distance import tf_nndistance
    a,b,c,d = tf_nndistance.nn_distance(p,q)
    cd1 = tf.reduce_mean(a)
    cd2 = tf.reduce_mean(c)
    return cd1+cd2


files = []
files_path = []

if(a.dataset == "shapenet"):
    f = open('./data/shapenet_val.txt','r')
    for index,line in enumerate(f):
        if(line.strip().split('/')[0]==a.class_idx):
            #print(line)
            files.append(line.strip().split('/')[1])
    f.close()

if(a.dataset == "famous"):
    f = open('./data/famous_testset.txt','r')
    for index,line in enumerate(f):
        #print(line)
        files.append(line.strip('\n'))
    f.close()
    
if(a.dataset == "ABC" or a.dataset == "other"):
    fileAll = os.listdir(INPUT_DIR)
    for file in fileAll:
        if(re.findall(r'.*.npz', file, flags=0)):
            print(file.strip().split('.')[0])
            files.append(file.strip().split('.')[0])

for file in files:
    files_path.append(INPUT_DIR + file + '.npz')
SHAPE_NUM = 1 # len(files_path)
print('SHAPE_NUM:',SHAPE_NUM)

pointclouds = []
samples = []
mm = 0
if(TRAIN):
    for file in files_path:
        load_data = np.load(file)
        gt = load_data['sample_near'][0]
        point = np.asarray(load_data['sample_near']).reshape(-1,POINT_NUM,3)
        sample = np.asarray(load_data['sample']).reshape(-1,POINT_NUM,3)
        pointclouds.append(point)
        samples.append(sample)
    
    pointclouds = np.asarray(pointclouds)
    samples = np.asarray(samples)
    print('data shape:',pointclouds.shape,samples.shape)
else:
    for file in files_path:
        load_data = np.load(file)
        points = np.asarray(load_data['sample_near']).reshape(-1, 3)
        pointclouds.append(points[:20000])
    pointclouds = np.asarray(pointclouds)

feature = tf.placeholder(tf.float64, shape=[BS,None,SHAPE_NUM])
points_target = tf.placeholder(tf.float64, shape=[BS,POINT_NUM,3])
input_points_3d = tf.placeholder(tf.float64, shape=[BS, POINT_NUM,3])
points_target_num = tf.placeholder(tf.int64, shape=[1,1])
points_input_num = tf.placeholder(tf.int64, shape=[1,1])
global_step = tf.placeholder(tf.float64, shape=[])
size_input = tf.placeholder(tf.float64, shape=[])
size_input3 = tf.placeholder(tf.float64, shape=[])
size_input2 = tf.placeholder(tf.float64, shape=[])

def smaller(p):
    p = (p - (VOX_SIZE - 1)/2.0) / ((VOX_SIZE-1)/ 2.0/bd)
    return p

def biliniear_interpolation_3d2(data, warp):
    """
    Interpolate a 3D array (monochannel).
    :param data: 3D tensor.
    :param warp: a list of 3D coordinates to interpolate. 2D tensor with shape (n_points, 3).
    """
    # warp =  (warp + 1) * (VOX_SIZE / 2)
    n_pts = warp.shape[0]
    # Pad data around to avoid indexing overflow
    # data = tf.pad(data, [[1, 1], [1, 1], [1, 1]], mode='SYMMETRIC')
    x, y, z = warp[:, 0], warp[:, 1], warp[:, 2]
    warp = tf.cast(warp, 'float64') #  + tf.constant([1, 1, 1], dtype='float64')
    i000 = tf.cast(tf.floor(warp), dtype=tf.int64)
    i100 = i000 + tf.constant([1, 0, 0], dtype=tf.int64)
    i010 = i000 + tf.constant([0, 1, 0], dtype=tf.int64)
    i001 = i000 + tf.constant([0, 0, 1], dtype=tf.int64)
    i110 = i000 + tf.constant([1, 1, 0], dtype=tf.int64)
    i101 = i000 + tf.constant([1, 0, 1], dtype=tf.int64)
    i011 = i000 + tf.constant([0, 1, 1], dtype=tf.int64)
    i111 = i000 + tf.constant([1, 1, 1], dtype=tf.int64)
    c000 = tf.gather_nd(data, i000)
    c100 = tf.gather_nd(data, i100)
    c010 = tf.gather_nd(data, i010)
    c001 = tf.gather_nd(data, i001)
    c110 = tf.gather_nd(data, i110)
    c101 = tf.gather_nd(data, i101)
    c011 = tf.gather_nd(data, i011)
    c111 = tf.gather_nd(data, i111)
    x0 = tf.cast(i000[:, 0], dtype=tf.float64)
    y0 = tf.cast(i000[:, 1], dtype=tf.float64)
    z0 = tf.cast(i000[:, 2], dtype=tf.float64)
    x1 = tf.cast(i111[:, 0], dtype=tf.float64)
    y1 = tf.cast(i111[:, 1], dtype=tf.float64)
    z1 = tf.cast(i111[:, 2], dtype=tf.float64)
    a0 = -(-c000*x1*y1*z1+c001*x1*y1*z0+c010*x1*y0*z1-c011*x1*y0*z0) - (c100*x0*y1*z1-c101*x0*y1*z0-c110*x0*y0*z1+c111*x0*y0*z0)
    a1 = -(c000*y1*z1-c001*y1*z0-c010*y0*z1+c011*y0*z0) - (-c100*y1*z1+c101*y1*z0+c110*y0*z1-c111*y0*z0)
    a2 = -(c000*x1*z1-c001*x1*z0-c010*x1*z1+c011*x1*z0) - (-c100*x0*z1+c101*x0*z0+c110*x0*z1-c111*x0*z0)
    a3 = -(c000*x1*y1-c001*x1*y1-c010*x1*y0+c011*x1*y0) - (-c100*x0*y1+c101*x0*y1+c110*x0*y0-c111*x0*y0)
    a4 = -(-c000*z1+c001*z0+c010*z1-c011*z0 + c100*z1 - c101*z0 - c110*z1 + c111*z0)
    a5 = -(-c000*y1+c001*y1+c010*y0-c011*y0 + c100*y1 - c101*y1 - c110*y0 + c111*y0)
    a6 = -(-c000*x1+c001*x1+c010*x1-c011*x1 + c100*x0 - c101*x0 - c110*x0 + c111*x0)
    a7 = -(c000-c001-c010+c011-c100+c101+c110-c111)

    f = a0 + a1 * x + a2 * y + a3 * z + a4 * x * y + a5 * x * z + a6 * y * z + a7 * x * y * z

    return f[:, None]
    
def biliniear_interpolation_3d(data, warp):
    """
    Interpolate a 3D array (monochannel).
    :param data: 3D tensor.
    :param warp: a list of 3D coordinates to interpolate. 2D tensor with shape (n_points, 3).
    """
    # warp =  (warp + 1) * (VOX_SIZE / 2)
    n_pts = warp.shape[0]
    # Pad data around to avoid indexing overflow
    data = tf.pad(data, [[1, 1], [1, 1], [1, 1]], mode='SYMMETRIC')
    warp = warp + tf.constant([1, 1, 1], dtype='float64')
    i000 = tf.cast(tf.floor(warp), dtype=tf.int64)
    i100 = i000 + tf.constant([1, 0, 0], dtype=tf.int64)
    i010 = i000 + tf.constant([0, 1, 0], dtype=tf.int64)
    i001 = i000 + tf.constant([0, 0, 1], dtype=tf.int64)
    i110 = i000 + tf.constant([1, 1, 0], dtype=tf.int64)
    i101 = i000 + tf.constant([1, 0, 1], dtype=tf.int64)
    i011 = i000 + tf.constant([0, 1, 1], dtype=tf.int64)
    i111 = i000 + tf.constant([1, 1, 1], dtype=tf.int64)
    c000 = tf.gather_nd(data, i000)
    c100 = tf.gather_nd(data, i100)
    c010 = tf.gather_nd(data, i010)
    c001 = tf.gather_nd(data, i001)
    c110 = tf.gather_nd(data, i110)
    c101 = tf.gather_nd(data, i101)
    c011 = tf.gather_nd(data, i011)
    c111 = tf.gather_nd(data, i111)
    # build matrix
    h00 = tf.ones(n_pts, dtype=tf.float64)
    x0 = tf.cast(i000[:, 0], dtype=tf.float64)
    y0 = tf.cast(i000[:, 1], dtype=tf.float64)
    z0 = tf.cast(i000[:, 2], dtype=tf.float64)
    x1 = tf.cast(i111[:, 0], dtype=tf.float64)
    y1 = tf.cast(i111[:, 1], dtype=tf.float64)
    z1 = tf.cast(i111[:, 2], dtype=tf.float64)
    h1 = tf.stack([h00, x0, y0, z0, x0 * y0, x0 * z0, y0 * z0, x0 * y0 * z0])
    h2 = tf.stack([h00, x1, y0, z0, x1 * y0, x1 * z0, y0 * z0, x1 * y0 * z0])
    h3 = tf.stack([h00, x0, y1, z0, x0 * y1, x0 * z0, y1 * z0, x0 * y1 * z0])
    h4 = tf.stack([h00, x1, y1, z0, x1 * y1, x1 * z0, y1 * z0, x1 * y1 * z0])
    h5 = tf.stack([h00, x0, y0, z1, x0 * y0, x0 * z1, y0 * z1, x0 * y0 * z1])
    h6 = tf.stack([h00, x1, y0, z1, x1 * y0, x1 * z1, y0 * z1, x1 * y0 * z1])
    h7 = tf.stack([h00, x0, y1, z1, x0 * y1, x0 * z1, y1 * z1, x0 * y1 * z1])
    h8 = tf.stack([h00, x1, y1, z1, x1 * y1, x1 * z1, y1 * z1, x1 * y1 * z1])
    h = tf.stack([h1, h2, h3, h4, h5, h6, h7, h8])
    h = tf.transpose(h, perm=[2, 0, 1])
    c = tf.transpose(tf.stack([c000, c100, c010, c110, c001, c101, c011, c111]))
    c = tf.expand_dims(c, -1)
    a = tf.matmul(tf.matrix_inverse(h), c)[:, :, 0]
    x = warp[:, 0]
    y = warp[:, 1]
    z = warp[:, 2]

    f = a[:, 0] + a[:, 1] * x + a[:, 2] * y + a[:, 3] * z + \
        a[:, 4] * x * y + a[:, 5] * x * z + a[:, 6] * y * z + a[:, 7] * x * y * z
    
    gradx = a[:, 1] + a[:, 4] * y + a[:, 5] * z + a[:, 7] * y * z
    grady = a[:, 2] + a[:, 4] * x + a[:, 6] * z + a[:, 7] * x * z
    gradz = a[:, 3] + a[:, 5] * x + a[:, 6] * y + a[:, 7] * x * y
    gradx = gradx[:, None]
    grady = grady[:, None]
    gradz = gradz[:, None]
    grad = tf.concat([gradx, grady, gradz], 1)

    return f[:, None], grad


def get_grad(data, points):
    points =  bigger(points) # (points + 1) * ((VOX_SIZE-1) / 2)
    err = size_input # global_step # SCALE # 1.0 /(VOX_SIZE) # dis ##第一处不同
    err2 = size_input
    err3 = size_input
    x_points = points + tf.concat([err * tf.ones([POINT_NUM, 1], 'float64'), tf.zeros([POINT_NUM, 2], 'float64')], 1)
    y_points = points + tf.concat([tf.zeros([POINT_NUM, 1], 'float64'), err2 * tf.ones([POINT_NUM, 1], 'float64'), tf.zeros([POINT_NUM, 1], 'float64')], 1)
    z_points = points + tf.concat([tf.zeros([POINT_NUM, 2], 'float64'), err3 * tf.ones([POINT_NUM, 1], 'float64')], 1)
    x_points_dev = points + tf.concat([-err * tf.ones([POINT_NUM, 1], 'float64'), tf.zeros([POINT_NUM, 2], 'float64')], 1)
    y_points_dev = points + tf.concat([tf.zeros([POINT_NUM, 1], 'float64'), -err2 * tf.ones([POINT_NUM, 1], 'float64'), tf.zeros([POINT_NUM, 1], 'float64')], 1)
    z_points_dev = points + tf.concat([tf.zeros([POINT_NUM, 2], 'float64'), -err3 * tf.ones([POINT_NUM, 1], 'float64')], 1)
    x_grad = biliniear_interpolation_3d(data, x_points) - biliniear_interpolation_3d(data, x_points_dev)
    y_grad = biliniear_interpolation_3d(data, y_points) - biliniear_interpolation_3d(data, y_points_dev)
    z_grad = biliniear_interpolation_3d(data, z_points) - biliniear_interpolation_3d(data, z_points_dev)
    grad = tf.concat([x_grad, y_grad, z_grad], 1)
    return grad


def get_special_vox(dim):
    a = sphere_init_sdf
    a = np.float64(a)
    a = tf.convert_to_tensor(a)
    vox_tensor = tf.Variable(a)
    return vox_tensor
    
    
def gridpull(input_points_3d, points_target):
    vox = get_special_vox(VOX_SIZE)
    points = bigger(input_points_3d, VOX_SIZE)  # (input_points_3d + 1) * (VOX_SIZE / 2)
    gtpoints = bigger(points_target, VOX_SIZE)
    sdf, grad = biliniear_interpolation_3d(vox, points[0])
    gtsdf, gtgrad = biliniear_interpolation_3d(vox, gtpoints[0])
    gtsdf = gtsdf[None]
    sdf = sdf[None]
    dis = tf.exp(tf.sqrt(tf.reduce_sum((input_points_3d - points_target)**2, 2)))
    grad = grad[None]
    gtgrad = gtgrad[None]
    normal_p_lenght = tf.expand_dims(safe_norm(grad, axis = -1), -1)
    gt_normal_p_length = tf.expand_dims(safe_norm(gtgrad, axis=-1), -1)
    grad_norm = grad /normal_p_lenght
    gtgrad_norm = gtgrad/gt_normal_p_length
    g_points = input_points_3d - sdf * grad_norm
    gtgrad2 = gtgrad 
    gtgrad2 = gtgrad2[None]
    gt_normal_p_length2 = tf.expand_dims(safe_norm(gtgrad2, axis=-1), -1)
    gtgrad_norm2 = gtgrad2/gt_normal_p_length2
    return g_points, sdf, vox, grad_norm, dis, gtsdf, gtgrad_norm, gtgrad_norm2, normal_p_lenght


def get_lr(step):
    step = tf.cast(step, tf.float32)
    n = tf.floor(tf.divide(step, 400))
    bilv = tf.pow(LR, n)
    lr = BASE_LR * bilv
    return lr


def get_sample_gt_pair3(gts, kdtree, batch_size, pn):
    index = np.random.choice(gts.shape[0], POINT_NUM_GT, replace = False)
    noise = gts[index[:POINT_NUM_GT // 2]] + np.random.uniform(-np.sqrt(3)/VOX_SIZE, np.sqrt(3)/VOX_SIZE,size=[gts.shape[0]//2, 3])
    noise2 = gts[index[POINT_NUM_GT // 2:]] + np.random.uniform(-0.002, 0.002, size=[gts.shape[0]//2, 3])
    noise = np.concatenate([noise, noise2], 0)
    dist, idx = kdtree.query(noise)
    select_gts = gts[idx]
    noise = noise[None]
    select_gts = select_gts[None]
    return noise, select_gts
    
    
def get_sample_gt_pair(gts, kdtree, batch_size, pn, step, sample_weight):
    noise = get_sample(gts, POINT_NUM_GT, SAMPLE_TYPE, step, sample_weight)
    dist, idx = kdtree.query(noise)
    select_gts = gts[idx]
    noise = noise[None]
    select_gts = select_gts[None]
    return noise, select_gts


g_points, sdf, voxs, grad, dis, gtsdf, gtgrad, gtgrad2, grad_len = gridpull(input_points_3d, points_target)

useful_index_tf = tf.cast(tf.convert_to_tensor(useful_index), tf.int32)
useful_index_tf_weight = tf.cast(tf.convert_to_tensor(useful_index_weight), tf.float64)
useful_middle_grid_sdf = tf.gather_nd(voxs, useful_index_tf)
vox_loss = tf.zeros(useful_middle_grid_sdf.shape, tf.float64)
for i in range(6):
    index = useful_index.copy()
    if i < 3:
        index[:, i] += 1
    else:
        index[:, i - 3] -= 1
    index = tf.cast(tf.convert_to_tensor(index), tf.int32)
    useful_grid_sdf = tf.gather_nd(voxs, index)
    vox_loss += (useful_grid_sdf - useful_middle_grid_sdf)**2

vox_loss = tf.reduce_mean(useful_index_tf_weight * vox_loss)

l2_loss =  tf.reduce_mean(dis * tf.norm((points_target-g_points), axis=-1))
sdf_loss = tf.reduce_mean((gtsdf)**2)
if TRAIN:
    sdf_mins_loss = tf.reduce_mean((grad[:, 3 * POINT_NUM_GT // 4 : 7 * POINT_NUM_GT // 8] - grad[:, 7 * POINT_NUM_GT // 8:])**2)
else:
    sdf_mins_loss = tf.zeros([], 'float64')


grad_len_loss = tf.reduce_mean((grad_len - 1)**2)
grad_loss = 1 - tf.reduce_mean(tf.reduce_sum(grad * gtgrad, 2))
print('l2_loss:',l2_loss)

vox_loss = VOX_LOSS_WEIGHT * vox_loss
grad_loss = GRAD_LOSS_WEIGHT * grad_loss
sdf_loss = SDF_LOSS_WEIGHT * sdf_loss

loss = l2_loss  + vox_loss + sdf_loss + grad_loss #+ grad_len_loss # + grad_loss #  + sdf_mins_loss # + 1e-3 * ll


t_vars = tf.trainable_variables()
optim = tf.train.AdamOptimizer(learning_rate=get_lr(global_step), beta1=0.9)
loss_grads_and_vars = optim.compute_gradients(loss, var_list=t_vars)
loss_optim = optim.apply_gradients(loss_grads_and_vars)

config = tf.ConfigProto(allow_soft_placement=False) 

saver_restore = tf.train.Saver(var_list=t_vars)
saver = tf.train.Saver(max_to_keep=2000000)


with tf.Session(config=config) as sess:
    feature_bs = []
    for i in range(SHAPE_NUM):
        tt = []
        for j in range(int(POINT_NUM)):
            t = np.zeros(SHAPE_NUM)
            t[i] = 1
            tt.append(t)
        feature_bs.append(tt)
    feature_bs = np.asarray(feature_bs)

    grid_points = []
    for i in range(VOX_SIZE):
        for j in range(VOX_SIZE):
            for k in range(VOX_SIZE):
                grid_points.append(np.array([i,j,k]))
    grid_points = np.asarray(grid_points)
    grid_points = smaller(grid_points)
   
    total_step = 4800
    if(TRAIN):
        print('train start')
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        
        POINT_NUM_GT_bs = np.array(POINT_NUM_GT).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        
        all_loss = []
        sample_weight = None
        sample_time = 0
        train_time = 0
        for i in range(total_step + 10):
            epoch_index = np.random.choice(SHAPE_NUM, SHAPE_NUM, replace = False)
            loss_i = 0
            at = time.time()
            noises, selectgts = get_sample_gt_pair(gttxt, kdtree, BS, POINT_NUM, i, sample_weight)
            sample_time += time.time() - at
            for epoch in range(1):
                input_points_2d_bs = noises[:, epoch * POINT_NUM : (epoch + 1) * POINT_NUM]
                point_gt = selectgts[:, epoch * POINT_NUM : (epoch + 1) * POINT_NUM]
                feature_bs_t = feature_bs[0,:,:].reshape(1,-1,SHAPE_NUM)
                if True: # i < 3000:
                    size_input_ss = np.random.rand() * 3
                    size_input2_ss = np.random.rand() * 3
                    size_input3_ss = np.random.rand() * 3
                else:
                    size_input_ss = np.random.rand() * 1
                    size_input2_ss = np.random.rand() * 1
                    size_input2_ss = np.random.rand() * 1
                at = time.time()
                _,loss_c, l2loss, voxloss, sdfloss, gradloss,  sdfs, ggp, grad_output, gtgrad_output, vv = sess.run([loss_optim,loss, l2_loss, vox_loss, sdf_loss, grad_loss, sdf, g_points, grad, gtgrad, voxs],feed_dict={size_input: size_input_ss, size_input2: size_input2_ss, size_input3: size_input3_ss, global_step: i, input_points_3d:input_points_2d_bs,points_target:point_gt,feature:feature_bs_t,points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
                train_time += time.time() - at
                loss_i = loss_i + loss_c
                # print(sdfs)
            loss_i = loss_i / SHAPE_NUM
            if(i%10 == 0):
                print('epoch:', i, 'epoch loss:', loss_i, 'l2 loss: ', l2loss, 'vox loss: ', voxloss,  'sdf loss: ', sdfloss, 'grad loss: ', gradloss)
            if(i%total_step == 0 and i > 10):
                print('save model')
                saver.save(sess, os.path.join(OUTPUT_DIR, "model"), global_step=i+1)
        end_time = time.time()
        print('time: ', train_time)
    else:
        print('test start')
        checkpoint = tf.train.get_checkpoint_state(OUTPUT_DIR).all_model_checkpoint_paths
        path = OUTPUT_DIR + 'model-' + str(INDEX * total_step + 1)
        print(path)
        saver.restore(sess, path)
        
        s = np.arange(-test_bd,test_bd, (2*test_bd)/TEST_VOX_SIZE)
            
        print(s.shape[0])
        vox_size = s.shape[0]
        POINT_NUM_GT_bs = np.array(vox_size).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        input_points_2d_bs = []
        for i in s:
            for j in s:
                for k in s:
                    input_points_2d_bs.append(np.asarray([i,j,k]))
        input_points_2d_bs = np.asarray(input_points_2d_bs)
        print('input_points_2d_bs',input_points_2d_bs.shape)
        input_points_2d_bs = input_points_2d_bs.reshape((vox_size,vox_size,vox_size,3))
        POINT_NUM_GT_bs = np.array(vox_size*vox_size).reshape(1,1)

        test_num = SHAPE_NUM
        print('test_num:',test_num)
        cd = 0
        nc = 0
        cd2 = 0
        for epoch in range(test_num):
            print('test:',epoch)
            vox = []
            voxgrad = []
            feature_bs = []
            for j in range(vox_size*vox_size):
                t = np.zeros(SHAPE_NUM)
                t[epoch] = 1
                feature_bs.append(t)
            feature_bs = np.asarray(feature_bs)
            for i in range(vox_size):
                input_points_2d_bs_t = input_points_2d_bs[i,:,:,:] 
                input_points_2d_bs_t = input_points_2d_bs_t.reshape(BS, vox_size*vox_size, 3)
                feature_bs_t = feature_bs.reshape(BS,vox_size*vox_size,SHAPE_NUM)
                size_input_ss = np.random.rand() * 4 + 1
                sdf_c = sess.run([sdf],feed_dict={size_input: size_input_ss, size_input3: size_input_ss, size_input2: size_input_ss, input_points_3d:input_points_2d_bs_t,feature:feature_bs_t,points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
                
                vox.append(sdf_c)
            vox = np.asarray(vox)
            vox = vox.reshape((vox_size,vox_size,vox_size))
            vox_max = np.max(vox.reshape((-1)))
            vox_min = np.min(vox.reshape((-1)))
            print('max_min:',vox_max,vox_min)
            
            threshs = [0, 0.001, 0.005] # , -0.001, -0.003, -0.005, -0.01, -0.02]
            for thresh in threshs:
                print(np.sum(vox>thresh),np.sum(vox<thresh))
                
                if(np.sum(vox>0.0)<np.sum(vox<0.0)):
                    thresh = -thresh
                print('model:',epoch,'thresh:',thresh)
                vertices, triangles = libmcubes.marching_cubes(vox, thresh)
                if(vertices.shape[0]<10 or triangles.shape[0]<10):
                    print('no sur---------------------------------------------')
                    continue
                if(np.sum(vox>0.0)>np.sum(vox<0.0)):
                    triangles_t = []
                    for it in range(triangles.shape[0]):
                        tt = np.array([triangles[it,2],triangles[it,1],triangles[it,0]])
                        triangles_t.append(tt)
                    triangles_t = np.asarray(triangles_t)
                else:
                    triangles_t = triangles
                    triangles_t = np.asarray(triangles_t)

                # vertices -= 0.5
                # Undo padding
                vertices -= 1
                # Normalize to bounding box
                vertices /= np.array([vox_size-1, vox_size-1, vox_size-1])
                vertices = 1.1 * (vertices - 0.5)

                vertices += 0.5
                vertices = vertices * (maxn - minn) + minn

                mesh = trimesh.Trimesh(vertices, triangles_t,
                               vertex_normals=None,
                               process=False)

                name = objn
                mesh.export(OUTPUT_DIR +  '/occn_' + name + '_'+ str(INDEX*100 + 1) + '_' + str(thresh) + '.off')
    
                mesh = trimesh.Trimesh(vertices, triangles,
                                   vertex_normals=None,
                                   process=False)
                # if(a.dataset == 'other'):
                    # continue
                if(a.dataset=="shapenet" or a.dataset=='other'):
                    ps, idx = mesh.sample(1000000, return_index=True)
                else:
                    ps, idx = mesh.sample(10000, return_index=True)
                ps = ps.astype(np.float32)
                normals_pred = mesh.face_normals[idx]
                
                if False: # (a.dataset=="shapenet" or a.dataset == 'other'):
                    data = np.load(GT_DIR + name + '/pointcloud.npz')
                    pointcloud = data['points']
                    normal = data['normals']
                else:
                    mesh_gt = trimesh.load(GT_DIR + name + '.ply')
                    pointcloud, idx_gt = mesh_gt.sample(10000, return_index=True)
                    pointcloud = pointcloud.astype(np.float32)
                    normal = mesh_gt.face_normals[idx_gt]
                
                nc_t,cd_t,cd2_t = eval_pointcloud(ps,pointcloud.astype(np.float32),normals_pred.astype(np.float32),normal.astype(np.float32))
                np.savez(OUTPUT_DIR + name + '_'+ str(thresh),pp = ps, np = normals_pred, p = pointcloud, n = normal, nc = nc_t, cd = cd_t, cd2 = cd2_t)
                nc = nc + nc_t
                cd = cd + cd_t
                cd2 = cd2 + cd2_t
        print('mean_nc:',nc/test_num,'mean_cd:',cd/test_num,'cd2:',cd2/test_num)
                    
    
    
