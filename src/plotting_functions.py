# Plotting functions for single dye


import matplotlib.collections as collections
import matplotlib.pyplot as plt
from classes import *


def initialise_figure(fig_size=(15, 5)):
    fig, ax = plt.subplots(figsize=fig_size)
    return fig, ax


def plot_sample_array(sample_array, plot_color="k"):
    plt.plot(np.linspace(0, len(sample_array) / 10, len(sample_array)), sample_array, plot_color)
    plt.ylim([0, max(sample_array[1000:]) * 1.2])


def plot_markers(dye_color: Dye, vertical=0, leftoffset=50):
    # make dict of loci present in chosen dye
    loci_on_dye = {locus_name: locus for (locus_name, locus) in locus_dict.items() if locus.dye == dye_color}
    newloclist = []
    newlabellist = []
    for (locus_name, locus) in loci_on_dye.items():
        plt.annotate(text="", xy=(locus.lower - leftoffset, vertical), xytext=(locus.upper - leftoffset, vertical),
                     arrowprops=dict(arrowstyle='<->', color='b'))
        newloclist.append((locus.lower + locus.upper) / 2 - leftoffset)
        newlabellist.append(locus_name)
    plt.xticks(newloclist, newlabellist)


def finish_plot(show_or_title="show"):
    if show_or_title == "show":
        plt.show()
    else:
        plt.savefig(show_or_title + ".png")
        plt.close()


def plot_peaks_analyst(peak_list: list, dye_color: Dye):
    for peak in peak_list:
        dye = peak.allele.dye
        if dye == dye_color:
            plt.plot([peak.allele.mid], [peak.height], str(dye.plot_color + "*"))


def plot_peaks_expected(peak_list: list, max_rel, dye_color: Dye):
    # amount to multiply relative peak height with
    plt.ylim([-50, max_rel])
    # plot max height relative points
    plt.hlines(max_rel, 0, 500)
    for peak in peak_list:
        dye = peak.allele.dye
        if dye == dye_color:
            color = dye.plot_color
            plt.plot([peak.allele.mid], [peak.height * max_rel], str(color + "*"))


def plot_analyst(peaks: list, sample: Sample):
    """uses both the analysts identified peaks and sized data \
    for comparison to plot both in one image for each color"""
    for j in range(6):
        current_plot = sample.data[:, j]
        current_dye = Dyes.color_list[j]
        initialise_figure()
        # plt.title(str('filename: '+sample.name+', dye: ' + str(Dyes.color_list[j].name)))
        plot_sample_array(current_plot)
        plot_markers(current_dye, 0)
        plot_peaks_analyst(peaks, current_dye)
        title = "Sample_" + sample.name + "_" + str(sample.replica) + "_analyst_peaks_" + current_dye.name
        finish_plot()


def plot_expected(peaks: list, sample: Sample):
    """uses both the theoretical actual relative peaks and \
    sized data for comparison to plot both in one image"""
    for j in range(6):
        # makes one separate figure per dye
        initialise_figure()
        # plt.title(str('filename: '+sample.name+', dye: ' + Dyes.color_list[j].name))
        current_plot = sample.data[:, j]
        current_dye = Dyes.color_list[j]
        max_relative = max(current_plot[1000:]) * 1.5
        plt.hlines(max_relative, 0, 500)
        plot_sample_array(current_plot)
        plot_peaks_expected(peaks, max_relative, current_dye)
        plot_markers(current_dye)
        finish_plot()  # "Sample_"+sample.name+"_"+str(sample.replica)+"_expected_peaks_"+current_dye.name)


def plot_labeled_line(sample_array, peak_bools):
    peaks = [sample_array[i] if peak_bools[i] else 0 for i in range(len(sample_array))]
    not_peaks = sample_array - peaks
    fig, ax = initialise_figure(fig_size=(25, 5))
    plot_sample_array(not_peaks, 'r')
    plot_sample_array(peaks, 'b')
    bottom, top = ax.get_ylim()
    plot_markers(Dyes.BLUE, bottom, 50)
    plt.show()
    plt.close(fig)


def plot_labeled_background(sample_array, peak_bools, dye_index):
    # note that sample_array and peak_bools have the same shape
    fig, ax = initialise_figure(fig_size=(25, 2))
    plot_sample_array(sample_array)
    x = np.linspace(0, len(sample_array) / 10, len(sample_array))
    # plot background of peaks in green
    y_min = -500
    y_max = 0.3 * max(sample_array[1000:])
    collection = collections.BrokenBarHCollection.span_where(x, ymin=y_min, ymax=y_max, where=peak_bools,
                                                             facecolor='green', alpha=0.5)
    ax.add_collection(collection)
    # plot non-peaks in red
    collection = collections.BrokenBarHCollection.span_where(x, ymin=y_min, ymax=y_max, where=~peak_bools,
                                                             facecolor='red', alpha=0.5)
    ax.add_collection(collection)
    # plot markers
    plt.ylim([y_min, y_max])
    plot_markers(Dyes.color_list[dye_index], y_min, 50)
    plt.show()
    plt.close(fig)


def plot_results_FFN(image, prediction, label, title="show"):
    fig, ax = initialise_figure(fig_size=(20, 3))
    image = image.squeeze()
    prediction = prediction.squeeze()
    label = label.squeeze()
    x = np.linspace(0, len(image) / 10, len(image))
    y_max = 1  # min(1000, 0.1 * max(input[:,dye]))
    y_min = 0  # -0.1*y_max      # always a 10% gap on bottom for legibility
    ax.set_xlim([0, 480])
    ax.set_ylim([y_min, y_max])
    ax.plot(x, image, "k")
    ax.axhline(y=0.5, linestyle="--", color="gray")
    # plot result
    ax.plot(x, prediction, color="magenta")
    # plot truth
    collection = collections.BrokenBarHCollection.span_where(x, ymin=y_min, ymax=y_max, where=label,
                                                             facecolor='cyan', alpha=0.3)
    ax.add_collection(collection)
    plot_markers(Dyes.BLUE, y_min, 50)
    finish_plot(title)


def boxplot_scores(dataframe, groupby: str, toplot: List):
    # print(df[df['upper'] == df['upper'].min()])
    dataframe.boxplot(toplot, groupby)
    finish_plot('show')


def scatterplot_scores_noc(dataframe):
    data_2 = dataframe[dataframe['noc'] == 2]
    data_3 = dataframe[dataframe['noc'] == 3]
    data_4 = dataframe[dataframe['noc'] == 4]
    data_5 = dataframe[dataframe['noc'] == 5]

    ax = data_2.plot.scatter(x='auc_FFN', y='auc_Unet', color='orange', marker='*')
    data_3.plot.scatter(x='auc_FFN', y='auc_Unet', color='red', marker='*', ax=ax)
    data_4.plot.scatter(x='auc_FFN', y='auc_Unet', color='purple', marker='*', ax=ax)
    data_5.plot.scatter(x='auc_FFN', y='auc_Unet', color='blue', marker='*', ax=ax)
    ax.set_xlim([0.93, 1])
    ax.set_ylim([0.93, 1])
    lims = [0.93, 1]
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.grid()
    ax.set_xlabel('AUC score FFN')
    ax.set_ylabel('AUC score U-net')
    plt.legend(['x=y', '2-donor', '3-donor', '4-donor', '5-donor'], loc='lower right')
    plt.title('AUC (ROC) of U-net against FFN')
    plt.show()

def scatterplot_scores_mix(dataframe):
    data_A = dataframe[dataframe['mix_type'] == 'A']
    data_B = dataframe[dataframe['mix_type'] == 'B']
    data_C = dataframe[dataframe['mix_type'] == 'C']
    data_D = dataframe[dataframe['mix_type'] == 'D']
    data_E = dataframe[dataframe['mix_type'] == 'E']

    ax = data_A.plot.scatter(x='auc_FFN', y='auc_Unet', color='yellow', marker='*')
    data_B.plot.scatter(x='auc_FFN', y='auc_Unet', color='darkorange', marker='*', ax=ax)
    data_C.plot.scatter(x='auc_FFN', y='auc_Unet', color='magenta', marker='*', ax=ax)
    data_D.plot.scatter(x='auc_FFN', y='auc_Unet', color='purple', marker='*', ax=ax)
    data_E.plot.scatter(x='auc_FFN', y='auc_Unet', color='blue', marker='*', ax=ax)
    ax.set_xlim([0.93, 1])
    ax.set_ylim([0.93, 1])
    lims = [0.93, 1]
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.grid()
    ax.set_xlabel('AUC score FFN')
    ax.set_ylabel('AUC score U-net')
    plt.legend(['x=y', 'mixture A', 'mixture B', 'mixture C', 'mixture D', 'mixture E'], loc='lower right')
    plt.title('AUC (ROC) of U-net against FFN')
    plt.show()