import tensorflow as tf
import numpy as np
import utils
import time
import networks
import traceback
import matplotlib.pyplot as plt

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim

from flags import FLAGS

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999



def get_model(netname):
    if netname == 'face':
        return networks.DNFaceMultiView('saved/weight.pkl')

if __name__ == '__main__':
    while True:
        try:
            if FLAGS.eval_dir == '':
                get_model(FLAGS.train_model).train()
            else:
                get_model(FLAGS.train_model).eval()

        except Exception as e:
            traceback.print_exc()
            time.sleep(10)
