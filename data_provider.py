import tensorflow as tf
import numpy as np
import menpo.io as mio
import menpo
import scipy
import utils

from pathlib import Path
from scipy.io import loadmat
from menpo.image import Image
from menpo.shape import PointCloud
from menpo.transform import Translation


def caffe_preprocess(image):
    VGG_MEAN = np.array([102.9801, 115.9465, 122.7717])
    # RGB -> BGR
    image = tf.reverse(image, [False, False, True])
    # Subtract VGG training mean across all channels
    image = image - VGG_MEAN.reshape([1, 1, 3])
    return image


def _rescale_image(image, stride_width=64, method=0):
    # make sure smallest size is 600 pixels wide & dimensions are (k * stride_width) + 1
    height = tf.to_float(tf.shape(image)[0])
    width = tf.to_float(tf.shape(image)[1])

    # Taken from 'szross'
    scale_up = 625. / tf.minimum(height, width)
    scale_cap = 961. / tf.maximum(height, width)
    scale_up = tf.minimum(scale_up, scale_cap)
    new_height = stride_width * tf.round(
        (height * scale_up) / stride_width) + 1
    new_width = stride_width * tf.round((width * scale_up) / stride_width) + 1
    new_height = tf.to_int32(new_height)
    new_width = tf.to_int32(new_width)
    image = tf.image.resize_images(
        image, (new_height, new_width), method=method)
    return image


def augment_img(img, augmentation):
    flip, rotate, rescale = np.array(augmentation).squeeze()
    rimg = img.rescale(rescale)
    rimg = rimg.rotate_ccw_about_centre(rotate)
    crimg = rimg.warp_to_shape(
        img.shape,
        Translation(-np.array(img.shape) / 2 + np.array(rimg.shape) / 2)
    )
    if flip > 0.5:
        crimg = crimg.mirror()

    img = crimg

    return img


def rotate_points_tensor(points, image, angle):

    s = tf.shape(image)
    image_center = tf.to_float(s[:2]) / 2.

    # center coordinates since rotation center is supposed to be in the image center
    points_centered = points - image_center

    rot_matrix = tf.dynamic_stitch([[0], [1], [2], [3]], [tf.cos(angle), -tf.sin(angle), tf.sin(angle), tf.cos(angle)])
    rot_matrix = tf.reshape(rot_matrix, shape=[2, 2])

    points_centered_rot = tf.matmul(rot_matrix, tf.transpose(points_centered))

    return tf.transpose(points_centered_rot) + image_center



def rotate_image_tensor(image, angle):
    s = tf.shape(image)
    image_center = tf.to_float(s[:2]) / 2.

    # Coordinates of new image
    xs, ys = tf.meshgrid(tf.range(0.,tf.to_float(s[1])), tf.range(0., tf.to_float(s[0])))
    coords_new = tf.reshape(tf.pack([ys,xs], 2), [-1, 2])

    # center coordinates since rotation center is supposed to be in the image center
    coords_new_centered = tf.to_float(coords_new) - image_center

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.pack(
        [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)])
    rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
    coord_old_centered = tf.matmul(
        rot_mat_inv, tf.transpose(coords_new_centered))
    coord_old = tf.to_int32(tf.round(
        tf.transpose(coord_old_centered) + image_center))


    # Find nearest neighbor in old image
    coord_old_y, coord_old_x = tf.unpack(coord_old, axis=1)


    # Clip values to stay inside image coordinates
    outside_y = tf.logical_or(tf.greater(
        coord_old_y, s[0]-1), tf.less(coord_old_y, 0))
    outside_x = tf.logical_or(tf.greater(
        coord_old_x, s[1]-1), tf.less(coord_old_x, 0))
    outside_ind = tf.logical_or(outside_y, outside_x)


    inside_mask = tf.logical_not(outside_ind)
    inside_mask = tf.tile(tf.reshape(inside_mask, s[:2])[...,None], tf.pack([1,1,s[2]]))

    coord_old_y = tf.clip_by_value(coord_old_y, 0, s[0]-1)
    coord_old_x = tf.clip_by_value(coord_old_x, 0, s[1]-1)
    coord_flat = coord_old_y * s[1] + coord_old_x

    im_flat = tf.reshape(image, tf.pack([-1, s[2]]))
    rot_image = tf.gather(im_flat, coord_flat)
    rot_image = tf.reshape(rot_image, s)


    return tf.select(inside_mask, rot_image, tf.zeros_like(rot_image))


def lms_to_heatmap(lms, h, w, n_landmarks, marked_index):
    xs, ys = tf.meshgrid(tf.range(0.,tf.to_float(w)), tf.range(0., tf.to_float(h)))
    sigma = 5.
    gaussian = (1. / (sigma * np.sqrt(2. * np.pi)))



    def gaussian_fn(lms):
        y, x, idx = tf.unpack(lms)
        idx = tf.to_int32(idx)
        def run_true():
            return tf.exp(-0.5 * (tf.pow(ys - y, 2) + tf.pow(xs - x, 2)) *
                   tf.pow(1. / sigma, 2.)) * gaussian * 17.

        def run_false():
            return tf.zeros((h,w))

        return tf.cond(tf.reduce_any(tf.equal(marked_index,idx)), run_true, run_false)


    img_hm = tf.pack(tf.map_fn(gaussian_fn, tf.concat(1, [lms, tf.to_float(tf.range(0,n_landmarks))[..., None]])))


    return img_hm


class ProtobuffProvider(object):
    def __init__(self, filename='train.tfrecords', root=None, batch_size=1, rescale=None, augmentation=False):
        self.filename = filename
        self.root = Path(root)
        self.batch_size = batch_size
        self.image_extension = 'jpg'
        self.rescale = rescale
        self.augmentation = augmentation


    def get(self):
        images, *names = self._get_data_protobuff(self.root / self.filename)
        tensors = [images]

        for name in names:
            tensors.append(name)

        return tf.train.shuffle_batch(
            tensors, self.batch_size, 1000, 200, 4)

    def augmentation_type(self):
        return tf.pack([tf.random_uniform([1]),
                        (tf.random_uniform([1]) * 60. - 30.) * np.pi / 180.,
                        tf.random_uniform([1]) * 0.5 + 0.75])

    def _get_features(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                # images
                'image': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                # svs
                'n_svs': tf.FixedLenFeature([], tf.int64),
                'n_svs_ch': tf.FixedLenFeature([], tf.int64),
                # 'svs_0': tf.FixedLenFeature([], tf.string),
                'svs_1': tf.FixedLenFeature([], tf.string),
                # 'svs_2': tf.FixedLenFeature([], tf.string),
                'svs_3': tf.FixedLenFeature([], tf.string),
                # landmarks
                'n_landmarks': tf.FixedLenFeature([], tf.int64),
                'gt': tf.FixedLenFeature([], tf.string),
                'visible': tf.FixedLenFeature([], tf.string),
                'marked': tf.FixedLenFeature([], tf.string),
                'scale': tf.FixedLenFeature([], tf.float32),
                # original infomations
                'original_scale': tf.FixedLenFeature([], tf.float32),
                'original_centre': tf.FixedLenFeature([], tf.string),
                'original_lms': tf.FixedLenFeature([], tf.string),
                # inverse transform to original landmarks
                'restore_translation': tf.FixedLenFeature([], tf.string),
                'restore_scale': tf.FixedLenFeature([], tf.float32)
            }

        )
        return features

    def _image_from_feature(self, features):
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image_height = tf.to_int32(features['height'])
        image_width = tf.to_int32(features['width'])
        #
        image = tf.reshape(image, (image_height, image_width, 3))
        image = tf.to_float(image)
        return image, image_height, image_width

    def _pose_from_feature(self, features):
        n_svs = tf.to_int32(features['n_svs'])
        n_svs_ch = tf.to_int32(features['n_svs_ch'])
        # svs_0 = tf.image.decode_jpeg(features['svs_0'])
        svs_1 = tf.image.decode_jpeg(features['svs_1'])
        # svs_2 = tf.image.decode_jpeg(features['svs_2'])
        svs_3 = tf.image.decode_jpeg(features['svs_3'])
        n_svs = 2
        image_height = tf.to_int32(features['height'])
        image_width = tf.to_int32(features['width'])

        pose = tf.reshape(tf.pack([svs_3, svs_1]),(n_svs,n_svs_ch,image_height,image_width))
        pose = tf.transpose(pose, perm=[2, 3, 0, 1])
        pose = tf.reshape(pose, (image_height,image_width,n_svs*n_svs_ch))
        pose = tf.to_float(pose) / 255.
        return pose, n_svs, n_svs_ch

    def _heatmap_from_feature(self, features):
        n_landmarks = tf.to_int32(features['n_landmarks'])
        gt_lms = tf.decode_raw(features['gt'], tf.float32)
        visible = tf.to_int32(tf.decode_raw(features['visible'], tf.int64))
        marked = tf.to_int32(tf.decode_raw(features['marked'], tf.int64))
        image_height = tf.to_int32(features['height'])
        image_width = tf.to_int32(features['width'])

        gt_lms = tf.reshape(gt_lms, (n_landmarks, 2))
        gt_heatmap = lms_to_heatmap(
            gt_lms, image_height, image_width, n_landmarks, marked)
        gt_heatmap = tf.transpose(gt_heatmap, perm=[1,2,0])

        return gt_heatmap, gt_lms, n_landmarks, visible, marked

    def _info_from_feature(self, features):
        scale = features['scale']
        return scale

    def _set_shape(self, image, pose, gt_heatmap, gt_lms):
        image.set_shape([None, None, 3])
        pose.set_shape([2, None, None, 7])
        gt_heatmap.set_shape([None, None, 16])
        gt_lms.set_shape([16, 2])

    def _flip_pose(self, pose, image_height,image_width,n_svs,n_svs_ch):
        pose = tf.reshape(pose, (image_height,image_width,n_svs,n_svs_ch))
        flip_pose_list = []
        for idx in [1,0,2,3,5,4,6]:
            flip_pose_list.append(pose[:,:,:,idx])
        pose = tf.pack(flip_pose_list, axis=3)
        pose = tf.reshape(pose, (image_height,image_width,n_svs*n_svs_ch))
        return pose

    # Data from protobuff
    def _get_data_protobuff(self, filename):
        filename = str(filename)
        filename_queue = tf.train.string_input_producer([filename],
                                                        num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = self._get_features(serialized_example)

        # image
        image, image_height, image_width = self._image_from_feature(features)

        # svs
        pose, n_svs, n_svs_ch = self._pose_from_feature(features)

        # landmarks
        gt_heatmap, gt_lms, n_landmarks, visible, marked = self._heatmap_from_feature(features)

        # infomations
        scale = self._info_from_feature(features)

        # augmentation
        if self.augmentation:
            do_flip, do_rotate, do_scale = tf.unpack(self.augmentation_type())

            # rescale
            image_height = tf.to_int32(tf.to_float(image_height) * do_scale[0])
            image_width = tf.to_int32(tf.to_float(image_width) * do_scale[0])

            image = tf.image.resize_images(image, tf.pack([image_height, image_width]))
            pose = tf.image.resize_images(pose, tf.pack([image_height, image_width]))
            gt_heatmap = tf.image.resize_images(gt_heatmap, tf.pack([image_height, image_width]))
            gt_lms *= do_scale


            # rotate
            image = rotate_image_tensor(image, do_rotate)
            pose = rotate_image_tensor(pose, do_rotate)
            gt_heatmap = rotate_image_tensor(gt_heatmap, do_rotate)
            gt_lms = rotate_points_tensor(gt_lms, image, do_rotate)


            # flip
            def flip_fn(image=image, pose=pose, gt_heatmap=gt_heatmap, gt_lms=gt_lms):
                image = tf.image.flip_left_right(image)
                pose = tf.image.flip_left_right(pose)
                gt_heatmap = tf.image.flip_left_right(gt_heatmap)

                pose = self._flip_pose(
                    pose, image_height,image_width,n_svs,n_svs_ch)

                flip_hm_list = []
                flip_lms_list = []
                for idx in [5,4,3,2,1,0,6,7,8,9,15,14,13,12,11,10]:
                    flip_hm_list.append(gt_heatmap[:,:,idx])
                    flip_lms_list.append(gt_lms[idx,:])

                gt_heatmap = tf.pack(flip_hm_list, axis=2)
                gt_lms = tf.pack(flip_lms_list)

                return image, pose, gt_heatmap, gt_lms

            def no_flip(image=image, pose=pose, gt_heatmap=gt_heatmap, gt_lms=gt_lms):
                return image, pose, gt_heatmap, gt_lms

            image, pose, gt_heatmap, gt_lms = tf.cond(do_flip[0] > 0.5, flip_fn, no_flip)

        # crop to 256 * 256
        target_h = tf.to_int32(256)
        target_w = tf.to_int32(256)
        offset_h = tf.to_int32((image_height - target_h) / 2)
        offset_w = tf.to_int32((image_width - target_w) / 2)

        image = tf.image.crop_to_bounding_box(
            image, offset_h, offset_w, target_h, target_w)

        pose = tf.image.crop_to_bounding_box(
            pose, offset_h, offset_w, target_h, target_w)
        pose = tf.reshape(pose, (target_h,target_w,n_svs,n_svs_ch))
        pose = tf.transpose(pose, perm=[2, 0, 1, 3])

        gt_heatmap = tf.image.crop_to_bounding_box(
            gt_heatmap, offset_h, offset_w, target_h, target_w)

        gt_lms -= tf.to_float(tf.pack([offset_h, offset_w]))

        self._set_shape(image, pose, gt_heatmap, gt_lms)

        return image, pose, gt_heatmap, gt_lms, scale
