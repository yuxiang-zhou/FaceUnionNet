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
            root=FLAGS.dataset_dir,
            batch_size=FLAGS.batch_size,
            rescale=FLAGS.rescale,
            augmentation=FLAGS.eval_dir=='',
            )

        image, pose, gt_heatmap, gt_lms, scale = provider.get()

        return image, pose, gt_heatmap, gt_lms, scale



    def _eval_matrix(self, lms_predictions, states, images, datas):
        *_, gt_landmarks, scales = datas

        def wrapper(lms_hm_prediction):
            bsize,h,w,n_ch = lms_hm_prediction.shape
            lms_hm_prediction_filter = np.stack(list(map(
                lambda x: scipy.ndimage.filters.gaussian_filter(*x),
                zip(lms_hm_prediction.transpose(0,3,1,2).reshape(-1,h,w), [3] * (bsize * n_ch)))))
            result = lms_hm_prediction_filter.reshape(
                bsize,n_ch,h,w).transpose(0,2,3,1)
            return np.array(result).astype(np.float32)

        lms_predictions, = tf.py_func(wrapper, [lms_predictions], [tf.float32])

        hs = tf.argmax(tf.reduce_max(lms_predictions, 2), 1)
        ws = tf.argmax(tf.reduce_max(lms_predictions, 1), 1)
        predictions = tf.transpose(tf.to_float(tf.pack([hs, ws])), perm=[1, 2, 0])

        return utils.pckh(predictions, gt_landmarks, scales)


    def _eval_summary_ops(self, accuracy, lms_predictions, states, images, datas):

        # These are streaming metrics which compute the "running" metric,
        # e.g running accuracy
        metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({

            "accuracy/pckh_All": slim.metrics.streaming_mean(accuracy[:,-1]),
            "accuracy/pckh_Head": slim.metrics.streaming_mean(accuracy[:,0]),
            "accuracy/pckh_Shoulder": slim.metrics.streaming_mean(accuracy[:,1]),
            "accuracy/pckh_Elbow": slim.metrics.streaming_mean(accuracy[:,2]),
            "accuracy/pckh_Wrist": slim.metrics.streaming_mean(accuracy[:,3]),
            "accuracy/pckh_Hip": slim.metrics.streaming_mean(accuracy[:,4]),
            "accuracy/pckh_Knee": slim.metrics.streaming_mean(accuracy[:,5]),
            "accuracy/pckh_Ankle": slim.metrics.streaming_mean(accuracy[:,6])
        })

        # Define the streaming summaries to write:
        summary_ops = []
        for metric_name, metric_value in metrics_to_values.items():
            op = tf.scalar_summary(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        summary_ops.append(tf.scalar_summary('accuracy/running_pckh', tf.reduce_mean(accuracy[:,-1])))
        summary_ops.append(tf.image_summary('predictions/landmark-regression', tf.reduce_sum(lms_predictions, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4)))
        summary_ops.append(tf.image_summary('images', images, max_images=min(FLAGS.batch_size,4)))

        return summary_ops, metrics_to_updates


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

                    # custom losses
                    self._build_losses(predictions, states, images, datas)

                    # total losses
                    total_loss = slim.losses.get_total_loss()
                    tf.scalar_summary('losses/total loss', total_loss)


                    # image summaries
                    self._build_summaries(predictions, states, images, datas)
                    tf.image_summary('images', images, max_images=min(FLAGS.batch_size,4))

                    # learning rate decay
                    global_step = slim.get_or_create_global_step()

                    learning_rate = tf.train.exponential_decay(
                        FLAGS.initial_learning_rate,
                        global_step,
                        FLAGS.learning_rate_decay_step / FLAGS.batch_size,
                        FLAGS.learning_rate_decay_factor,
                        staircase=True)

                    tf.scalar_summary('learning rate', learning_rate)

                    optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

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


    def eval(self):

        g = tf.Graph()

        with g.as_default():
            # Load datasets.
            images, *datas = self._get_data()
            images /= 255.

            # Define model graph.
            with tf.variable_scope('net'):
                with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                    is_training=False):

                    lms_predictions, states = self._build_network(images)


        with tf.Session(graph=g) as sess:

            accuracy = self._eval_matrix(lms_predictions, states, images, datas)
            # These are streaming metrics which compute the "running" metric,
            # e.g running accuracy
            summary_ops, metrics_to_updates = self._eval_summary_ops(
                accuracy, lms_predictions, states, images, datas)

            global_step = slim.get_or_create_global_step()
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
                summary_op=tf.merge_summary(summary_ops),
                eval_interval_secs=30)



class DNCatFace(DeepNetwork):
    """docstring for DeepNetwork"""
    def __init__(self, path, output_lms=FLAGS.n_landmarks):
        super(DNCatFace, self).__init__(output_lms=output_lms)
        self.network_path = path

    def _build_network(self, inputs):

        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        with open(self.network_path, 'br') as f:
            data = pickle.load(f, encoding='latin1')

        subnet_regression = data['children'][2].copy()
        subnet_regression['children'][-2]['children'].pop()
        subnet_regression['children'].pop()

        net = slim.conv2d(
            inputs,
            self.output_lms + 3,
            1,
            activation_fn=None
        )

        net = utils.build_graph(net, subnet_regression)

        net = slim.conv2d(
            net,
            self.output_lms,
            1,
            activation_fn=None
        )
        net = slim.conv2d_transpose(
            net,
            self.output_lms,
            4,
            4,
            activation_fn=None,
            padding='VALID'
        )

        return net, []

    def _build_losses(self, predictions, states, images, datas):
        heatmap, *_ = datas

        # landmark-regression losses
        weight_hm = utils.get_weight(heatmap, tf.ones_like(heatmap), ng_w=0.1, ps_w=1) * 500
        l2norm = slim.losses.mean_squared_error(predictions, heatmap, weight=weight_hm)
        tf.scalar_summary('losses/lms_pred', l2norm)

    def _build_summaries(self, predictions, states, images, datas):
        heatmap, *_ = datas

        tf.image_summary('predictions/landmark-regression', tf.reduce_sum(predictions, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))
        # tf.image_summary('gt/all ', tf.reduce_sum(heatmap * tf.ones(heatmap), -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))

    def _get_data(self):
        provider = data_provider.CatFaceProvider(
            root=FLAGS.dataset_dir,
            batch_size=FLAGS.batch_size,
            rescale=FLAGS.rescale,
            augmentation=FLAGS.eval_dir=='',
            )

        image, gt_heatmap, gt_lms, scale = provider.get()

        return image, gt_heatmap, gt_lms, scale
