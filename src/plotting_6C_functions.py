# Plotting functions for all 6 dyes in one figure


import matplotlib.collections as collections
from matplotlib import pyplot as plt
import numpy as np
from classes import *


def plot_inputs_PROVEDIt(result: np.array, title: str, leftoffset=50, fig_size=(25, 15), Dyes=None):
    # wanted to check out the data without labels
    number_of_dyes = 6
    fig, axes = plt.subplots(nrows=number_of_dyes, figsize=fig_size)
    result = result.squeeze()
    x_array = np.linspace(0, len(result) / 10, len(result))
    for dye in range(number_of_dyes):
        y_max = 1
        y_min = -0.1 * y_max  # always a 10% gap on bottom for legibility
        axes[dye].set_ylim([y_min, y_max])
        plot_markers(Dyes.color_list[dye], axes[dye], locus_dict_alt, y_min, leftoffset)
        axes[dye].plot(x_array, result[:, dye], "k")
        axes[dye].set_xlim([0, 480])
    plt.title(title)
    plt.show()
    plt.close()


def plot_results_unet_against_truth(input, result, label, title=False, leftoffset=50, fig_size=(30, 20)):
    number_of_dyes = 6
    fig, axes = plt.subplots(nrows=number_of_dyes, figsize=fig_size)
    input = input.squeeze()
    result = result.squeeze()
    x_array = np.linspace(0, len(result) / 10, len(result))
    for dye in range(number_of_dyes):
        y_max = 1  # min(1000, 0.1 * max(input[:,dye]))
        y_min = -0.1 * y_max  # always a 10% gap on bottom for legibility
        axes[dye].set_xlim([0, 480])
        axes[dye].set_ylim([y_min, y_max])
        plot_markers(Dyes.color_list[dye], axes[dye], locus_dict_alt, y_min, leftoffset)
        axes[dye].plot(x_array, input[:, dye], "k")
        axes[dye].axhline(y=500, linestyle="--", color="gray")
        # plot result
        line1, = axes[dye].plot(x_array, y_max * result[:, dye], color="magenta")
        # plot truth
        plot_labels(input[:, dye], label[:, dye], axes[dye], y_min, y_max, alph=0.3, nopeak="w", peak="c")
        ax_right = axes[dye].twinx()
        ax_right.set_ylim([-0.1, 1])
        ax_right.spines["right"].set_color(line1.get_color())
        ax_right.tick_params(axis='y', colors=line1.get_color())
    if not title:
        plt.show()
    else:
        plt.savefig(str(title) + ".png")
    plt.close()


def plot_results_unet_against_truth_alt(input, result, label, title=False, leftoffset=50, fig_size=(30, 20)):
    number_of_dyes = 6
    fig, axes = plt.subplots(nrows=number_of_dyes, figsize=fig_size)
    input = input.squeeze()
    result = result.squeeze()
    x_array = np.linspace(0, len(result) / 10, len(result))
    for dye in range(number_of_dyes):
        y_max = 1000  # min(1000, 0.1 * max(input[:,dye]))
        y_min = -0.1 * y_max  # always a 10% gap on bottom for legibility
        axes[dye].set_xlim([0, 480])
        axes[dye].set_ylim([y_min, y_max])
        plot_markers(Dyes.color_list[dye], axes[dye], locus_dict, y_min, leftoffset)
        axes[dye].plot(x_array, input[:, dye], "k")
        # plot result
        line1, = axes[dye].plot(x_array, y_max * result[:, dye])
        # plot truth
        collection = collections.BrokenBarHCollection.span_where(x_array, ymin=y_min, ymax=y_max,
                                                                 where=label[:, dye],
                                                                 facecolor='green', alpha=0.5)
        axes[dye].add_collection(collection)
        # plot result
        collection = collections.BrokenBarHCollection.span_where(x_array, ymin=y_min, ymax=y_max,
                                                                 where=(result > 0.5)[:, dye],
                                                                 facecolor='blue', alpha=0.3)
        axes[dye].add_collection(collection)
        ax_right = axes[dye].twinx()
        ax_right.set_ylim([-0.1, 1])
        ax_right.spines["right"].set_color(line1.get_color())
        ax_right.tick_params(axis='y', colors=line1.get_color())
    if not title:
        plt.show()
    else:
        plt.savefig(str(title) + ".png")
    plt.close()


def plot_inputs_unet(input, label, leftoffset=50, fig_size=(30, 20), rescale=10, title=''):
    number_of_dyes = 6
    fig, axes = plt.subplots(nrows=number_of_dyes, figsize=fig_size)
    input = input.squeeze()
    x_array = np.linspace(0, len(input) / rescale, len(input))
    plt.suptitle(title)
    for dye in range(number_of_dyes):
        y_max = 1000  # min(1000, 0.1 * max(input[:,dye]))
        y_min = -0.1 * y_max  # always a 10% gap on bottom for legibility
        axes[dye].set_ylim([y_min, y_max])
        plot_markers(Dyes.color_list[dye], axes[dye], locus_dict_alt, y_min, leftoffset)
        axes[dye].plot(x_array, input[:, dye], "k")
        # axes[dye].set_xlim([0,480])
        # plot truth
        plot_labels(input[:, dye], label[:, dye], axes[dye], y_min, y_max, rescale=rescale)
    plt.show()
    plt.close()


def plot_labels(sample_array, peak_bools, ax, y_min, y_max, alph=0.5, nopeak="purple", peak="green", rescale=10):
    # note that sample_array and peak_bools have the same shape
    x = np.linspace(0, len(sample_array) / rescale, len(sample_array))
    # plot background of peaks in green
    collection = collections.BrokenBarHCollection.span_where(x, ymin=y_min, ymax=y_max, where=peak_bools,
                                                             facecolor=peak, alpha=alph)
    ax.add_collection(collection)
    # plot background of non-peaks in red
    collection = collections.BrokenBarHCollection.span_where(x, ymin=y_min, ymax=y_max, where=~peak_bools,
                                                             facecolor=nopeak, alpha=alph)
    ax.add_collection(collection)


def plot_markers(dye_color: Dye, ax: object, dyekit: {}, vertical: object = 0.0, leftoffset: object = 50) -> object:
    # make dict of loci present in chosen dye
    loci_on_dye = {locus_name: locus for (locus_name, locus) in dyekit.items() if locus.dye == dye_color}
    newticklist = []
    newlabellist = []
    for (locus_name, locus) in loci_on_dye.items():
        ax.annotate(text="", xy=(locus.lower - leftoffset, vertical), xytext=(locus.upper - leftoffset, vertical),
                    arrowprops=dict(arrowstyle='<->', color='b'))
        newticklist.append((locus.lower + locus.upper) / 2 - leftoffset)
        newlabellist.append(locus_name)
    ax.set_xticks(newticklist)
    ax.set_xticklabels(newlabellist)


def plot_bins_vs_labels(input, labels_peaks, labels_bins, title=False, leftoffset=50, fig_size=(30, 20)):
    number_of_dyes = 6
    fig, axes = plt.subplots(nrows=number_of_dyes, figsize=fig_size)
    input = input.squeeze()
    x_array = np.linspace(0, len(input) / 10, len(input))
    for dye in range(number_of_dyes):
        y_max = 20000  # min(1000, 0.1 * max(input[:,dye]))
        y_min = -0.1 * y_max  # always a 10% gap on bottom for legibility
        axes[dye].set_ylim([y_min, y_max])
        plot_markers(Dyes.color_list[dye], axes[dye], locus_dict, y_min, leftoffset)
        line1, = axes[dye].plot(x_array, input[:, dye], "k")

        # plot background of peaks in green
        collection = collections.BrokenBarHCollection.span_where(x_array, ymin=y_min, ymax=y_max,
                                                                 where=labels_peaks[:, dye],
                                                                 facecolor='green', alpha=0.3)
        axes[dye].add_collection(collection)
        # plot background of bins in blue
        collection = collections.BrokenBarHCollection.span_where(x_array, ymin=y_min, ymax=y_max,
                                                                 where=labels_bins[500:5300, dye],
                                                                 facecolor='blue', alpha=0.3)
        axes[dye].add_collection(collection)
        axes[dye].set_xlim([0, 100])
    if not title:
        plt.show()
    else:
        plt.savefig(str(title) + ".png")
    plt.close()


def plot_Unet_against_CNN(input_for_nn, Unet_output, CNN_output, title, rescale, fig_size=(10, 15)):
    number_of_dyes = 6
    fig, axes = plt.subplots(nrows=number_of_dyes, figsize=fig_size)
    input_for_nn = input_for_nn.squeeze()
    Unet_output = Unet_output.squeeze()
    CNN_output = CNN_output.squeeze()

    x_array = np.linspace(0, len(Unet_output) / 10, len(Unet_output))
    x_array_cnn = np.linspace(0 + 25, len(CNN_output) / rescale + 25, len(CNN_output))
    for dye in range(number_of_dyes):
        y_max = 1.1
        y_min = -0.1  # * y_max  # always a 10% gap on bottom for legibility
        axes[dye].set_ylim([y_min, y_max])
        plot_markers(Dyes.color_list[dye], axes[dye], locus_dict_alt, y_min, 0)
        axes[dye].plot(x_array, Unet_output[:, dye], "m", alpha=0.3)
        axes[dye].plot(x_array_cnn, CNN_output[:, dye], "c", alpha=0.3)
        axes[dye].plot(x_array, input_for_nn[:, dye], "k", alpha=0.7)
        axes[dye].set_xlim([0, 480])
    plt.suptitle(title)
    plt.savefig(title + "_UnetCNN.png")
    plt.close()
