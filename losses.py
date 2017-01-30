import tensorflow as tf
slim = tf.contrib.slim


def smooth_l1(pred, ground_truth, weights=1):
    """Defines a robust L1 loss.

    This is a robust L1 loss that is less sensitive to outliers
    than the traditional L2 loss. This was defined in Ross
    Girshick's Fast R-CNN, ICCV 2015 paper.

    Args:
      pred: A `Tensor` of dimensions [num_images, height, width, 3].
      ground_truth: A `Tensor` of dimensions [num_images, height, width, 3].
      weights: A `Tensor` of dimensions [num_images,] or a scalar with the
          weighting per image.
    Returns:
      A scalar with the mean loss.
    """
    residual = tf.abs(pred - ground_truth)

    loss = tf.select(tf.less(residual, 1),
                     0.5 * tf.square(residual),
                     residual - .5)

    return slim.losses.compute_weighted_loss(loss, weights)


def quaternion_loss(pred, ground_truth, weights=None):
    '''Computes the half cosine distance between two quaternions.
    
    Assumes that the quaternions are l2 normalised.

    Args:
      pred: A `Tensor` of dimensions [num_images, 4]
      ground_truth: A `Tensor` of dimensions [num_images, 4].
    Returns:
      A scalar with the mean cosine loss.
    '''
    loss = 1 - tf.abs(tf.reduce_sum(pred * ground_truth, 1))

    return slim.losses.compute_weighted_loss(loss, weights)


def cosine_loss(pred, ground_truth, weights=None, dim=3):
    '''Computes the cosine distance between two images.
    
    Assumes that the input images are l2 normalised per pixel.

    Args:
      pred: A `Tensor` of dimensions [num_images, height, width, 3]
      ground_truth: A `Tensor` of dimensions [num_images, height, width, 3].
    Returns:
      A scalar with the mean angular error (cosine loss).
    '''
    loss = 1 - tf.reduce_sum(pred * ground_truth, dim)
    return slim.losses.compute_weighted_loss(loss, weights)

def normalized_rmse(pred, gt_truth, weights=1):
    '''Computes the error normalised by the interocular distance.
    
    Args:
      pred: A `Tensor` of dimensions [num_images, 68, 2].
      ground_truth: A `Tensor` of dimensions [num_images, 68, 2].
    Returns:
      A scalar with the normalized rmse error.
    '''
    norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, 36, :] - gt_truth[:, 45, :])**2), 1))
    loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * 68)
    return slim.losses.compute_weighted_loss(loss, weights)