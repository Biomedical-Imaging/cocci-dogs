import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from glob import glob
from model import auc
from keras.models import load_model
from vis.visualization import overlay
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import load_img
from vis.visualization.saliency import visualize_cam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold, KFold

np.random.seed(0)


def load_hp_results(prefix='src/', trial=None):
    file_name = prefix + 'Results/sherpa_results_new.txt'

    df = pd.read_csv(file_name)
    # df = df.iloc[24412:].reset_index(drop=True)
    df = df[df['Trial-ID'] != 'Trial-ID']

    for c in df.columns:
        try: df[c] = pd.to_numeric(df[c])
        except: pass
    
    if trial is not None:
        return df[df['Trial-ID'] == trial]
    
    return df


def load_dog(img_path):
    img = np.array(load_img(img_path))

    if img.shape[0] > img.shape[1]:
        img = np.array(
            load_img(img_path, target_size=(500, 400, 3)),
        ).transpose(1, 0, 2)
    else:
        img = np.array(
            load_img(img_path, target_size=(400, 500, 3)),
        )

    return (np.maximum(img, 0) / img.max()) * 255.0


def load_dogs(path):
    images_in_dir = []
    paths_in_dir  = []
    for img_path in glob(path):

        if img_path.lower().endswith('.jpg') or img_path.lower().endswith('.png'):
            img = load_dog(img_path)

            images_in_dir.append(img)
            paths_in_dir.append(img_path)

    return images_in_dir, paths_in_dir


# def load_data(prefix='/baldig/physicstest/'):
#     # LOAD IMAGES FROM DIRECTORY
#     normal, normal_paths = load_dogs(prefix + 'ValleyFeverDogs/HealthyCropped/*')
#     normal1, normal_paths1 = load_dogs(prefix + 'ValleyFeverDogs/Healthy/*')
#
#     cocci, cocci_paths = load_dogs(prefix + 'ValleyFeverDogs/DiseaseCropped/*')
#     cocci1, cocci_paths1 = load_dogs(prefix + 'ValleyFeverDogs/Disease/*')
#
#     # print(len(cocci), cocci_paths)
#     # COMBINE NORMAL + COCCI - SET TARGETS
#     X = np.array(
#         normal + normal1 + cocci + cocci1
#     )
#     Y = np.array(
#         [[1, 0]] * len(normal) + [[1, 0]] * len(normal1) +
#         [[0, 1]] * len(cocci) + [[0, 1]] * len(cocci1)
#     )
#     df_paths = pd.DataFrame({
#         'path': normal_paths + normal_paths1 + cocci_paths + cocci_paths1
#     })
#
#     idxs = np.array(range(len(X)))
#     np.random.shuffle(idxs)
#
#     print('Data:', X.shape, Y.shape)
#
#     return X[idxs] / 255., Y[idxs], df_paths.iloc[idxs]


def load_data(prefix='/baldig/physicstest/'):
    # LOAD IMAGES FROM DIRECTORY
    # normal, normal_paths = load_dogs(prefix + 'ValleyFeverDogs/HealthyCropped/*')
    # normal1, normal_paths1 = load_dogs(prefix + 'ValleyFeverDogs/Healthy/*')
    #
    # cocci , cocci_paths  = load_dogs(prefix + 'ValleyFeverDogs/DiseaseCropped/*')
    # cocci1 , cocci_paths1 = load_dogs(prefix + 'ValleyFeverDogs/Disease/*')
    normal1, normal_paths1 = load_dogs('../data/HealthyCropped/*')
    normal, normal_paths = [], []

    cocci1, cocci_paths1 = load_dogs('../data/DiseaseCropped/*')
    cocci, cocci_paths = [], []

    # COMBINE NORMAL + COCCI - SET TARGETS
    X = np.array(
        normal + normal1 + cocci + cocci1
    )
    Y = np.array(
        [[1,0]] * len(normal) + [[1,0]] * len(normal1) +
        [[0,1]] * len(cocci) + [[0,1]] * len(cocci1)
    )
    df_paths = pd.DataFrame({
        'path': normal_paths + normal_paths1 + cocci_paths + cocci_paths1
    })
    
    idxs = np.array(range(len(X)))
    np.random.shuffle(idxs)

    print('Data:', X.shape, Y.shape)
    unique_paths = pd.DataFrame({
        'path' : normal_paths1 + cocci_paths1
    })
    unique_labels = np.array(
        [[1,0]] * len(normal1) + [[0,1]] * len(cocci1)
    )
    print(unique_labels, unique_paths)
    return X[idxs] / 255., Y[idxs], df_paths.iloc[idxs], (unique_paths, unique_labels)


def get_same_name(inputs, labels, all_paths, unique_paths, idxs):
    I = []
    P = []

    for idx in idxs:
        count = 0

        path = '_'.join(
            os.path.basename(
                unique_paths.iloc[idx].path
            ).split('_')[:-1]
        )

        healthy = 'Healthy' in unique_paths.iloc[idx].path

        for i in range(len(all_paths)):
            p = all_paths.iloc[i].path

            if path in p and (healthy == ('Healthy' in p)):
                count += 1
                I.append(i)
                # X.append(inputs[i])
                # Y.append(labels[i])
                P.append(p)

            if count == 2:
                break

    return inputs[I], labels[I], pd.DataFrame({'path':P})


def cross_validation(args):
    X, Y, df_paths, (unique_paths, unique_labels) = load_data()

    # DETERMINE TYPE OF CROSS VALIDATION
    if args['cross_val'] == 'stratified':
        kf    = StratifiedKFold(n_splits=args['num_folds'], random_state=42)
        split = [[[0], [1]]] # kf.split(unique_paths, np.argmax(unique_labels, -1))
    else:
        kf    = KFold(n_splits=args['num_folds'], random_state=42)
        split = kf.split(X)

    for fold, (train_index, test_index) in enumerate(split):
        if args['notsherpa'] and args['sherpa_trial'] == 92 and fold > 5:
            continue

        x_train, y_train, p_train = get_same_name(X, Y, df_paths, unique_paths, train_index)
        x_test, y_test, p_test = get_same_name(X, Y, df_paths, unique_paths, test_index)

        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        print(y_train.sum(axis=0), y_test.sum(axis=0))

        # for i in range(2):
        #     print(p_train.iloc[i].path)

        yield fold, (x_train, x_test, y_train, y_test), (p_train, p_test)


# def cross_validation(args):
#     X, Y, df_paths = load_data()
#
#     # DETERMINE TYPE OF CROSS VALIDATION
#     if args['cross_val'] == 'stratified':
#         kf    = StratifiedKFold(n_splits=args['num_folds'], random_state=42)
#         split = kf.split(X, np.argmax(Y, -1))
#     else:
#         kf    = KFold(n_splits=args['num_folds'], random_state=42)
#         split = kf.split(X)
#
#     for fold, (train_index, test_index) in enumerate(split):
#
#         # TRAIN TEST SPLIT FOR FOLD
#         x_train, x_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
#
#         train_files = df_paths.iloc[train_index]
#         for i in range(5):
#             print(train_files.iloc[i].path)
#         yield fold, (x_train, x_test, y_train, y_test), (df_paths.iloc[train_index], df_paths.iloc[test_index])


def gen_samples(args, x_train, x_test, y_train, y_test):
    if args['data_aug']:
        gen = ImageDataGenerator(
            featurewise_center=args['normalize'],
            featurewise_std_normalization=args['normalize'],
            # rotation_range=10,
            # horizontal_flip=True,
            # vertical_flip=False
        )

        if args['normalize']:
            gen.fit(x_train)

        def get_gen_images(X, Y):
            generated_images = []; generated_labels = []
            gen_flow = gen.flow(X, Y, shuffle=False, batch_size=1)

            for i, (x, y) in enumerate(gen_flow):
                if i == len(X): break
                generated_images.append(x)
                generated_labels.append(y)

            generated_images = np.vstack(generated_images)
            generated_labels = np.concatenate(generated_labels)
            assert np.all(generated_labels == Y)
            return generated_images

        test_images = get_gen_images(x_test, y_test)
        train_images = get_gen_images(x_train, y_train)

        return train_images, test_images

    else:
        return x_train, x_test


def cam(args):
    # img_path = '/baldig/physicstest/ValleyFeverDogs/Healthy/EE544610_cropped.jpg'
    # img = load_dog(img_path)
    #
    # x_train = 1 - (np.array([img]) / 255.)
    # y_train = np.array([[1, 0]])
    # p_train = pd.DataFrame({
    #     'path' : [img_path]
    # })
    #
    # x_test  = np.array([img]) / 255.
    # y_test  = np.array([[1, 0]])
    # p_test  = pd.DataFrame({
    #     'path' : [img_path]
    # })

    fold  = int(args['model_path'].split('/')[-1].replace('.h5', '')) - 1
    model = load_model(args['model_path'], custom_objects={'auc': auc})

    for fold_num, data, (p_train, p_test) in cross_validation(args):
        if fold_num == fold: break

    x_train, x_test, y_train, y_test = data
    x_train_samples, x_test_samples = gen_samples(args, x_train, x_test, y_train, y_test)

    test_probabilities  = model.predict(x_test_samples)
    train_probabilities = model.predict(x_train_samples)

    test_predictions = np.argmax(test_probabilities, axis=1)
    test_targets     = np.argmax(y_test, axis=1)
    test_acc = accuracy_score(test_targets, test_predictions)

    train_predictions = np.argmax(train_probabilities, axis=1)
    train_targets = np.argmax(y_train, axis=1)
    train_acc = accuracy_score(train_targets, train_predictions)
    print(
        'Train ACC:', train_acc, np.sum(train_targets) / float(len(train_targets)),
        'Test ACC:', test_acc, np.sum(test_targets) / float(len(test_targets))
    )

    os.makedirs('../results/CAM/%d' % args['sherpa_trial'], exist_ok=True)

    def plot(predictions, targets, images, original_images, path_info, prefix=''):
        for i in range(len(targets)):
            # if i < 140: continue
            file_path = '../results/CAM/{}/{}{}'.format(args['sherpa_trial'], prefix, i)

            cam = visualize_cam(
                model,
                len(model.layers) - 1,
                predictions[i],
                images[i][np.newaxis]
            )

            plt.subplot(1, 2, 1)
            plt.imshow(original_images[i])
            plt.title('Original Image'); plt.tight_layout(); plt.axis('off')

            plt.subplot(1,2, 2)
            plt.imshow(
                overlay(cam, (original_images[i]) * 255.)
            )
            plt.title('Heat Map'); plt.tight_layout(); plt.axis('off')

            plt.suptitle(path_info.iloc[i]['path'].replace('/baldig/physicstest/ValleyFeverDogs/', ''))
            plt.savefig(file_path)

    print(len(train_predictions), len(train_targets), len(x_train_samples), len(x_train))
    plot(test_predictions, test_targets, x_test_samples, x_test, p_test, prefix='test_')
    plot(train_predictions, train_targets, x_train_samples, x_train, p_train, prefix='train_')


def plot_confusion_matrix(cm, args=None, prefix='', classes=['Healthy', 'Cocci'], ax=None):
    cmap = plt.cm.Blues
    title = 'Average Confusion Matrix for 10 Folds'

    if ax is None: fig, ax = plt.subplots(figsize=(15,15)); print(cm)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
#            title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.ylim(1.5, -0.5)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    if args is not None: fig.savefig('Models/%d/%scm.png' % (args.sherpa_trial, prefix))
