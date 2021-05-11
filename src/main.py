import os
import fcntl
import pprint
import sherpa
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob
from filelock import FileLock
from model import Network, auc
from keras.models import load_model
from utils import cross_validation, cam, load_hp_results, load_data, gen_samples

############## REPRODUCIBILITY ############
tf.set_random_seed(0)
np.random.seed(0)
###########################################

parser = argparse.ArgumentParser()

# MODEL DEFAULTS
parser.add_argument('--optimizer', default='adam')                       # WHAT OPT TO USE
parser.add_argument('--model_type', default='vgg')                       # VGG OR SHALLOW
parser.add_argument('--weights', default='imagenet')                     # PRETRAINED WEIGHTS
parser.add_argument('--lr', default=0.001, type=float)                   # LEARNING RATE
parser.add_argument('--noise', default=0.1, type=float)                  # NOISE AUG ON INPUT
parser.add_argument('--dropout', default=0., type=float)                 # DROPOUT
parser.add_argument('--lr_decay', default=1., type=float)                # LEARNING RATE DECAY

parser.add_argument('--num_dense_layers', default=1, type=int)           # NUMBER OF DENSE LAYERS
parser.add_argument('--num_dense_nodes', default=32, type=int)           # NUMBER OF NODES

parser.add_argument('--num_conv_layers',  default=4, type=int)           # NUMBER OF CONV LAYERS
parser.add_argument('--num_conv_filters', default=64, type=int)          # NUMBER OF CONV FILTERS

parser.add_argument('--train_base', action='store_true', default=0)      # TRAIN THE BASE OF VGG

# RUN SINGLE TRIAL WITHOUT HP SEARCH
parser.add_argument('--notsherpa', action='store_true')                  # DEFAULTS TO USING SHERPA
parser.add_argument('--sherpa_trial', default=-1, type=int)              # LOAD TRIAL FROM SHERPA RES

# PLOTTING
parser.add_argument('--cm', action='store_true', default=0)              # CONFUSION MATRIX
parser.add_argument('--cam', action='store_true', default=0)             # CLASS ACTIVATION MAPPING

# TRAINING
parser.add_argument('--gpu', default=-1)
parser.add_argument('--verbose', default=2, type=int)                    # PRINT TRAINING UPDATES
parser.add_argument('--epochs', default=100, type=int)                   # NUM EPOCHS
parser.add_argument('--data_aug', default=0, type=int)                   # USING KERAS DATA GEN
parser.add_argument('--normalize', default=0, type=int)                  # USING KERAS DATA GEN
parser.add_argument('--patience', default=15, type=int)                  # PATIENCE FOR EARLY STOPPING
parser.add_argument('--num_folds', default=10, type=int)                 # NUM FOLDS CROSS VAL
parser.add_argument('--cross_val', default='stratified')                 # TYPE OF CROSS VAL SPLIT
parser.add_argument('--batch_size', default=16, type=int)                # BATCH SIZE
parser.add_argument('--class_weight', default=0, type=int)               # WEIGHT IMBALANCED CLASSES
parser.add_argument('--model_path', default='Models/00001.h5',type=str)  # PATH FOR SAVING MODEL
FLAGS = parser.parse_args()

###### CONTROL TF GROWTH AND LOGGING ######
if FLAGS.notsherpa: gpu = FLAGS.gpu
else: gpu = os.environ.get('SHERPA_RESOURCE', '')
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto(); config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
###########################################

args = vars(FLAGS)
pp = pprint.PrettyPrinter(indent=4)

if FLAGS.notsherpa:
    class Trial: id = FLAGS.sherpa_trial

    client = None;
    trial = Trial()
    result_file_name = 'Results/results_%d.txt' % FLAGS.sherpa_trial

    if FLAGS.sherpa_trial != -1:
        os.makedirs('Models/%d' % FLAGS.sherpa_trial, exist_ok=True)
        trial_res = load_hp_results(prefix='', trial=trial.id)

        print('Loaded settings from trial:', FLAGS.sherpa_trial)

        for k in args:
            if k not in ['gpu', 'model_path', 'cam', 'cm']: args[k] = trial_res[k].unique()[0]

    args['notsherpa'] = 1
    args['sherpa_trial'] = trial.id
    pp.pprint(args)

if FLAGS.cam:
    print('Creating CAM heatmaps')
    cam(args)

elif FLAGS.cm:
    from sklearn.metrics import confusion_matrix
    from utils import plot_confusion_matrix

    train_cm = np.zeros((2,2))
    test_cm  = np.zeros((2, 2))

    for fold, data, path_info in cross_validation(vars(FLAGS)):
        model = load_model('Models/%d/%05d.h5' % (FLAGS.sherpa_trial, fold+1), custom_objects={'auc':auc})

        x_train, x_test, y_train, y_test = data

        train, test = path_info
        train['split'] = 'train'
        test['split'] = 'test'

        x_train_samples, x_test_samples, = gen_samples(args, x_train, x_test, y_train, y_test)
        test_probabilities = model.predict(x_test_samples)
        train_probabilities = model.predict(x_train_samples)

        predictions = np.argmax(train_probabilities, axis=1)
        targets = np.argmax(y_train, axis=1)
        train_cm += confusion_matrix(targets, predictions)
        train['prediction'] = predictions
        train['target'] = targets
        train['prob'] = train_probabilities[:, 1]

        predictions   = np.argmax(test_probabilities, axis=1)
        targets       = np.argmax(y_test, axis=1)
        test_cm += confusion_matrix(targets, predictions)
        test['prediction'] = predictions
        test['target'] = targets
        test['prob'] = test_probabilities[:, 1]

        df = train.append(
            test,
            ignore_index=True
        )
        print('SAVING TO:', 'Models/%d/%05d.csv' % (FLAGS.sherpa_trial, fold+1), df.shape)
        df.to_csv('Models/%d/%05d.csv' % (FLAGS.sherpa_trial, fold+1))

    plot_confusion_matrix(train_cm / 10., FLAGS, prefix='train_')
    plot_confusion_matrix(test_cm / 10., FLAGS, prefix='test_')

else:

    # USING SHERPA
    if not FLAGS.notsherpa:

        client = sherpa.Client()
        trial = client.get_trial()

        # UPDATE DEFAULTS WITH TRIAL
        args.update(trial.parameters)

        result_file_name = 'Results/sherpa_results_new.txt'

        pp.pprint(args)

    fold_results = {'auc':[], 'loss':[], 'acc': [], 'val_auc':[], 'val_loss':[], 'val_acc':[]}
    for fold, data, path_info in cross_validation(args):
        print('Fold:', fold+1)
        # BUILD NETWORK
        net = Network(args)

        # CALCULATE TRAIN TEST SPLITS
        _, _, train, test = data
        train_percent = train.sum(0) / float(len(train))
        test_percent = test.sum(0) / float(len(test))

        # TRAINING
        history = net.train(data, fold=fold, client=client, trial=trial)
        # print(history)

        # RECORDING RESULTS
        df = pd.DataFrame(history)

        if client is not None:
            for k in fold_results:
                if 'loss' in k:
                    fold_results[k].append(np.min(history[k]))
                else:
                    fold_results[k].append(np.max(history[k]))

        df['Trial-ID']    = trial.id
        df['fold']        = fold + 1
        df['epoch']       = range(1,df.shape[0]+1)
        df['train_split'] = np.max(train_percent)
        df['test_split']  = np.max(test_percent)

        for k, v in args.items():
            if k != 'gpus': df[k] = v

        columns = sorted(df.columns)
        format_str = ','.join('{}' for _ in columns) + '\n'

        logging.disable(logging.CRITICAL)

        with FileLock(result_file_name + ".lock"):
            with open(result_file_name, 'a+') as g:

                if not g.readlines():
                    g.write(','.join(columns) + '\n')

                for idx, row in df.iterrows():
                    g.write(
                        format_str.format(
                            *row[columns].values
                        )
                    )

        logging.disable(logging.NOTSET)

        if args['notsherpa']:
            print(
                'Fold: {}, Loss: {}, Val Loss: {}, Acc: {}, Val Acc: {}, AUC: {}, Val AUC: {}, Train: {}, Test: {}'.format(
                    fold+1,
                    round(np.min(history['loss']),2),
                    round(np.min(history['val_loss']),2),
                    round(np.max(history['acc']),2),
                    round(np.max(history['val_acc']),2),
                    round(np.max(history['auc']), 2),
                    round(np.max(history['val_auc']), 2),
                    round(np.max(train_percent),2),
                    round(np.max(test_percent),2)
            ))
        else:
            if fold == 3:
                break

    if not args['notsherpa']:
        for k in fold_results:
            fold_results[k] = float(np.mean(fold_results[k]))

        client.send_metrics(trial, 0, fold_results['val_auc'], context=fold_results)