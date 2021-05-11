import os
import sherpa
import argparse
import datetime
import itertools

# from utils import build_directory

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default=os.environ['CUDA_VISIBLE_DEVICES'])
parser.add_argument('--max_concurrent', type=int, default=4, help='Number of concurrent processes')

parser.add_argument('--max_layers', type=int, default=2)
FLAGS = parser.parse_args()

parameters = [
    sherpa.Choice('batch_size', [8]),

    sherpa.Continuous('lr', [0.0001, 0.01]),
    sherpa.Continuous('lr_decay', [0.5, 1.0]),
    sherpa.Choice('optimizer', ['rmsprop', 'adam', 'sgd']),

    # WHAT TYPE OF MODEL TO USE
    sherpa.Choice('model_type', ['shallow', 'vgg', 'inception', 'resnet', 'mobilenet']),

    # SHARED MODEL PARAMS
    sherpa.Discrete('num_dense_layers', [0, FLAGS.max_layers]),
    sherpa.Discrete('num_dense_nodes', [32, 256]),
    sherpa.Continuous('dropout', [0, 0.5]),
    sherpa.Choice('noise', [0.1]),

    # SHALLOW MODEL PARAMS
    # sherpa.Discrete('num_conv_layers', [1, FLAGS.max_layers]),
    # sherpa.Choice('num_conv_filters', [8, 16, 32, 64, 128]),

    # VGG MODEL
    sherpa.Choice('train_base', [0, 1]), # 1
    sherpa.Choice('weights', ['none', 'imagenet']),

    # USING KERAS DATA GENERATOR FOR AUG
    sherpa.Choice('data_aug', [1]),
    sherpa.Choice('normalize', [1]),

    # LOSS PENALTY FOR CLASS IMBALANCE
    sherpa.Choice('class_weight', [0, 1]),
]

for k, v in vars(FLAGS).items():
    parameters.append(
        sherpa.Choice(k, [v])
    )

# Run on local machine.
gpus = [int(x) for x in FLAGS.gpus.split(',')]
processes_per_gpu = FLAGS.max_concurrent // len(gpus)
assert FLAGS.max_concurrent % len(gpus) == 0
resources = list(itertools.chain.from_iterable(itertools.repeat(x, processes_per_gpu) for x in gpus))

sched = sherpa.schedulers.LocalScheduler(resources=resources)
alg = sherpa.algorithms.RandomSearch(max_num_trials=1000)
# alg = sherpa.algorithms.GPyOpt(max_num_trials=1000)

sherpa.optimize(
    parameters=parameters,
    algorithm=alg,
    lower_is_better=False,
    command='/pkg/python/3.6.1-centos7/bin/python3.6 main.py',
    scheduler=sched,
    max_concurrent=FLAGS.max_concurrent,
    output_dir='SherpaResults'
)
