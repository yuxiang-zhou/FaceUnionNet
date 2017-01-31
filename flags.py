import tensorflow as tf

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                          '''Initial learning rate.''')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          '''Epochs after which learning rate decays.''')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97,
                          '''Learning rate decay factor.''')
tf.app.flags.DEFINE_float('learning_rate_decay_step', 30000,
                          '''Learning rate decay factor.''')
tf.app.flags.DEFINE_float('pckat', 0.5,
                          '''PCK Measurement @''')

tf.app.flags.DEFINE_integer('batch_size', 1, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('eval_size', 500, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_iterations', 2, '''The number of iterations to unfold the pose machine.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            '''How many preprocess threads to use.''')
tf.app.flags.DEFINE_integer('n_landmarks', 38,
                            '''number of landmarks''')
tf.app.flags.DEFINE_integer('rescale', 256,
                            '''training mode''')

tf.app.flags.DEFINE_string('train_model', '',
                            '''training mode''')
tf.app.flags.DEFINE_string('dataset_dir', '/vol/atlas/databases/body/SupportVectorBody/crop340-mpii-train/',
                           '''Directory where to load datas '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('eval_dir', '',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')
tf.app.flags.DEFINE_string(
    'pretrained_resnet_checkpoint_path', '',
    '''If specified, restore this pretrained resnet '''
    '''before beginning any training.'''
    '''This restores only the weights of the resnet model''')
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            '''Number of batches to run.''')
tf.app.flags.DEFINE_string('train_device', '/gpu:0',
                           '''Device to train with.''')

tf.app.flags.DEFINE_string('pred_mode', 'pred',
                            '''prediction mode''')
tf.app.flags.DEFINE_string('db_name', 'mpii',
                            '''db name''')
tf.app.flags.DEFINE_integer('flip_pred', 0,
                            '''db name''')
