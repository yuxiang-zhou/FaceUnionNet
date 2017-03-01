import tensorflow as tf
import numpy as np
import losses
import data_provider
import utils
import matplotlib.pyplot as plt
import pickle
import scipy

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim

from flags import FLAGS

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999




def restore_resnet(sess, path):
    def name_in_checkpoint(var):
        # Uncomment for non lightweight model
        # if 'resnet_v1_50/conv1/weights' in var.name:
        #     return None
        name = '/'.join(var.name.split('/')[2:])
        name = name.split(':')[0]
        if 'Adam' in name:
            return None
        return name

    variables_to_restore = slim.get_variables_to_restore(
        include=["net/multiscale/resnet_v1_50"])
    variables_to_restore = {name_in_checkpoint(var): var
                            for var in variables_to_restore if name_in_checkpoint(var) is not None}

    return slim.assign_from_checkpoint_fn(path, variables_to_restore, ignore_missing_vars=True)


# general framework
class DeepNetwork(object):
    """docstring for DeepNetwork"""
    def __init__(self, output_lms=FLAGS.n_landmarks):
        super(DeepNetwork, self).__init__()
        self.output_lms = output_lms

    def _build_network(self, inputs):
        pass


    def _build_losses(self, predictions, states, images, datas):
        pass


    def _build_summaries(self, predictions, states, images, datas):
        pass


    def _build_restore_fn(self, sess):
        init_fn = None

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model ...')
            variables_to_restore = slim.get_model_variables()
            init_fn =  slim.assign_from_checkpoint_fn(
                FLAGS.pretrained_model_checkpoint_path,
                variables_to_restore,
                ignore_missing_vars=True)
        return init_fn


    def _get_data(self):
        provider = data_provider.ProtobuffProvider(
            filename=FLAGS.dataset_dir,
            batch_size=FLAGS.batch_size,
            rescale=FLAGS.rescale,
            augmentation=FLAGS.eval_dir=='',
            )

        image, pose, gt_heatmap, gt_lms, scale = provider.get()

        return image, pose, gt_heatmap, gt_lms, scale


    def train(self):
        g = tf.Graph()
        logging.set_verbosity(10)


        with g.as_default():
            # Load datasets.

            images, *datas = self._get_data()
            images /= 255.

            # Define model graph.
            with tf.variable_scope('net'):
                with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                is_training=True):

                    predictions, states = self._build_network(images)
                    # predictions = tf.identity(predictions, name='output')

                    # custom losses
                    self._build_losses(predictions, states, images, datas)

                    # total losses
                    total_loss = slim.losses.get_total_loss()
                    tf.summary.scalar('losses/total_loss', total_loss)

                    # learning rate decay
                    global_step = slim.get_or_create_global_step()

                    learning_rate = tf.train.exponential_decay(
                        FLAGS.initial_learning_rate,
                        global_step,
                        FLAGS.learning_rate_decay_step / FLAGS.batch_size,
                        FLAGS.learning_rate_decay_factor,
                        staircase=True)

                    optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

            # summaries
            tf.summary.image('images', images[..., :3], max_outputs=min(FLAGS.batch_size,4))
            tf.summary.scalar('learning_rate', learning_rate)
            self._build_summaries(predictions, states, images, datas)

        with tf.Session(graph=g) as sess:
            init_fn = self._build_restore_fn(sess)
            train_op = slim.learning.create_train_op(
                total_loss,
                optimizer,
                summarize_gradients=True)

            logging.set_verbosity(1)

            slim.learning.train(train_op,
                FLAGS.train_dir,
                save_summaries_secs=60,
                init_fn=init_fn,
                save_interval_secs=600)


    def _eval_matrix(self, lms_predictions, states, images, datas):
        pass


    def _eval_summary_ops(self, accuracy, lms_predictions, states, images, datas):
        pass


    def eval(self):

        with tf.Graph().as_default() as g:
            # Load datasets.
            images, *datas = self._get_data()
            images /= 255.

            # Define model graph.
            with tf.variable_scope('net'):
                with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                    is_training=False):

                    lms_predictions, states = self._build_network(images)
                    global_step = slim.get_or_create_global_step()


        with tf.Session(graph=g) as sess:

            accuracy = self._eval_matrix(lms_predictions, states, images, datas)
            # These are streaming metrics which compute the "running" metric,
            # e.g running accuracy
            summary_ops, metrics_to_updates = self._eval_summary_ops(
                accuracy, lms_predictions, states, images, datas)

            # Evaluate every 30 seconds
            logging.set_verbosity(1)

            # num_examples = provider.num_samples()
            # num_batches = np.ceil(num_examples / FLAGS.batch_size)
            # num_batches = 500

            slim.evaluation.evaluation_loop(
                '',
                FLAGS.train_dir,
                FLAGS.eval_dir,
                num_evals=FLAGS.eval_size,
                eval_op=list(metrics_to_updates.values()),
                summary_op=tf.summary.merge(summary_ops),
                eval_interval_secs=30)




class DNFaceMultiView(DeepNetwork):
    """docstring for DeepNetwork"""
    def __init__(self, path, output_lms=FLAGS.n_landmarks):
        super().__init__(output_lms=output_lms)
        self.network_path = path

    def _build_network(self, inputs):

        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        states = []

        with open(self.network_path, 'br') as f:
            data = pickle.load(f, encoding='latin1')

        subnet_regression = data['children'][2].copy()
        subnet_regression['children'][-2]['children'].pop()
        subnet_regression['children'].pop()


        net = slim.conv2d(
            inputs,
            77 + 3,
            1,
            activation_fn=None
        )

        net = utils.build_graph(net, subnet_regression)

        net = slim.conv2d(
            net,
            77,
            1,
            activation_fn=None
        )
        net = slim.conv2d_transpose(
            net,
            77,
            4,
            4,
            activation_fn=None,
            padding='VALID'
        )

        states.append(net)

        return net, states

    def _build_losses(self, predictions, states, images, datas):
        gt_heatmap, gt_lms, mask_index, pts_mask = datas

        # landmark-regression-39 losses
        weight_hm = utils.get_weight(gt_heatmap, tf.ones_like(gt_heatmap), ng_w=0.1, ps_w=1) * 500
        weight_hm *= pts_mask[:,None,None,:]

        l2norm = slim.losses.mean_squared_error(predictions, gt_heatmap, weights=weight_hm)

        tf.summary.scalar('losses/lms_pred', l2norm)

    def _build_summaries(self, predictions, states, images, datas):
        gt_heatmap, gt_lms, mask_index, pts_mask = datas

        tf.summary.image('predictions/landmark-regression', tf.reduce_sum(predictions, -1)[...,None] * 255.0, max_outputs=min(FLAGS.batch_size,4))

    def _get_data(self):
        provider = data_provider.ProtobuffProvider(
            filename=FLAGS.dataset_dir,
            batch_size=FLAGS.batch_size,
            rescale=FLAGS.rescale,
            augmentation=FLAGS.eval_dir=='',
            )

        return provider.get()
