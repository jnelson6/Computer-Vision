"""
Julia Nelson
Homework 3
CS-558
I pledge my honor that I have abided by the Stevens Honor System.
"""
import os
import numpy as np
import pandas as pd
from matplotlib import image
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

rgb_index = {'r': 0, 'g': 1, 'b': 2}
colors = {'r': 'RED', 'g': 'GREEN', 'b': 'BLUE'}


def scan_images(imgdirs):
    """
    Scan images and parse the file names in a list of source directories
    :param imgdirs: A list of directories
    :return: A list of images with parsed information
    """
    imglist = []
    imglist_test = []
    imglist_train = []
    # imdirs = ['ImClass/']
    for imdir in imgdirs:
        for filename in os.listdir(imdir):
            label, role = filename[:-5].split('_')
            id = int(filename.split('.')[0][-1])
            if role == 'train':
                imglist_train.append({'label': label,
                            'filename': filename,
                            'dir': imdir,
                            'role': role,
                            'id': id})
            else:
                imglist_test.append({'label': label,
                            'filename': filename,
                            'dir': imdir,
                            'role': role,
                            'id': id})
            imglist.append({'label': label,
                            'filename': filename,
                            'dir': imdir,
                            'role': role,
                            'id': id})
    # print("Training images:\n{}\n\n".format(imglist_train))
    # print("Testing images:\n{}".format(imglist_test))
    return imglist


def compute_multihist(imglist, bins, savefig=False):
    """
    Computes color histogram for each image in the set
    :param imglist: A list of images
    :param bins: Number of bins
    :param savefig: The histogram figure is saved in a file if savefig=True
    :return: Computed Data
    """
    bins_names = generate_bins_names(bins)
    # print(bins_names)
    df_header = {bin_name: [] for bin_name in bins_names}
    df_header['role'] = []
    df_header['label'] = []
    df_header['id'] = []
    # print(df_header)
    df_img = pd.DataFrame(df_header)
    for i, img in enumerate(imglist):
        img_data = image.imread(os.path.join(img['dir'], img['filename']))
        # print("{:02d} - Processing image ({} bins): {} ...".format(i, bins, img['filename']))
        # print("     Class:", img['label'])
        # print("     Role: ", img['role'])
        # print("     Shape: {}".format(img_data.shape))
        hists = np.empty(0)
        for color, index in rgb_index.items():
            hist, bins_ranges = np.histogram(img_data[:, :, index], bins=bins)
            hists = np.concatenate((hists, hist))
        df_img = df_img.append(pd.Series(hists, bins_names), ignore_index=True)
        df_img.loc[i, ['role', 'label', 'id']] = [img['role'], img['label'], img['id']]
    if savefig:
        save_histfig(df_img, bins)
    return df_img


def save_histfig(df_img, bins):
    """
    Save histogram figures to image files within 'plots/' directory
    :param df_img: Data of color histogram
    :param bins: Number of bins
    :return: None
    """
    bins_names_sep = generate_bins_names_sep(bins)
    bins_ranges = np.arange(1, bins + 1) * 255 / bins
    # print(bins_ranges, len(bins_ranges))
    for i, row in df_img.iterrows():
        for c, bins_names in bins_names_sep.items():
            plt.plot(
                bins_ranges,
                row[bins_names], c,
                label='{} Histogram'.format(colors[c]))
            plt.legend()
            plt.xlabel('Pixel values (0-255)')
            plt.ylabel('Number of pixels')
            try:
                os.mkdir('plots')
            except:
                pass
            plt.savefig('plots/bins_{:02d}_{}_{}_{}.jpg'.format(
                bins, row['label'], row['role'], int(row['id'])
            ))
        plt.clf()


def classify(df_img, bins, k):
    """
    Classify the data of color histogram for a given bins number and k nearest neighbor
    :param df_img: Data of color histogram
    :param bins: Number of bins
    :param k: Number of nearest neighbor
    :return: Classified data and test accuracy
    """
    bins_names = generate_bins_names(bins)
    X_train, Y_train, X_test, Y_test = split_train_test(df_img, bins_names)
    classifer = KNeighborsClassifier(algorithm='brute', n_neighbors=k)
    classifer.fit(X_train, Y_train['label'])
    pred_test = classifer.predict(X_test)
    acc_test = compute_accuracy(Y_test['label'], pred_test)
    return Y_test, pred_test, acc_test


def split_train_test(df_img, bins_names):
    """
    Split the data into training/testing data based on the filename
    :param df_img: Dataframe of images histograms
    :param bins_names: Set of bins names
    :return: Split data
    """
    # Split training data
    X_train = df_img[df_img['role'] == 'train']
    Y_train = X_train[['label', 'id']]
    X_train = X_train[bins_names]

    # Split testing data
    X_test = df_img[df_img['role'] == 'test']
    Y_test = X_test[['label', 'id']]
    X_test = X_test[bins_names]
    return X_train, Y_train, X_test, Y_test


def generate_bins_names(bins=8):
    """
    Generates a set of bins names for the dataset
    :param bins: Number of bins
    :return: Set of bins names for r, g, b
    """
    return ['{}_{:02d}'.format(color, bin_index)
            for color in rgb_index
            for bin_index in range(bins)]


def generate_bins_names_sep(bins=8):
    """
    Generates a set of separate bins names for the dataset
    :param bins: Number of bins
    :return: Set of bins names
    """
    r_bins = ['{}_{:02d}'.format('r', index) for index in range(bins)]
    g_bins = ['{}_{:02d}'.format('g', index) for index in range(bins)]
    b_bins = ['{}_{:02d}'.format('b', index) for index in range(bins)]
    return {'r': r_bins, 'g': g_bins, 'b': b_bins}


def compute_accuracy(actual, predicted):
    """
    Computes the test accuracy given a test sample with actual and predicted labels
    :param actual: Actual labels
    :param predicted: Predicted Labels
    :return: Test Accuracy (float)
    """
    matches = 0
    for i, act in enumerate(actual):
        if act == predicted[i]:
            matches += 1
    return matches / len(actual)
