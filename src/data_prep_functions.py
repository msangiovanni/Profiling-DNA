# All functions applied to data to make input for networks
# 'alt' is often used to distinguish between a different dye kit


import numpy as np
from reading_functions import make_person_mixture, OLD_make_person_mix_PROVEDIt
from classes import *


def create_DTDP_inputs_from_sample(sample: Sample, width=100, range_start=500, range_end=5300):
    """For one electropherogram, creates all input (node) images and their labels.
    @type sample: object
    """
    # width is amount of steps in each direction, usually 100
    sample_data = sample.data
    window_list = []
    labels = find_peaks_flowing_out_of_bins(sample_data, sample.name)
    label_list = []
    # iterate over all windows with center in same range as U-net
    for window_index in range(range_start, range_end):
        center_location = window_index
        left = center_location - width
        right = center_location + width + 1
        window = sample_data[left: right, :]

        flat_window = window.flatten('F')  # flatten it icw DTDP, F means dyes first (column-major)
        window_list.append(flat_window)
        label = labels[center_location, 0]
        label_list.append(label)
        # uncomment if you also want non-blue dyes
        # for dye in range(1, number_of_dyes):
        #     label = labels[center_location, dye]
        #     label_list.append(label)
        #     to_end = window[:,:1]                     # put different dye first
        #     start = window[:,1:]
        #     new_window = np.append(start, to_end)
        #     flat_window = new_window.flatten('F')           # flatten it icw DTDP, F means dyes first (column-major)
        #     window_list.append(flat_window)
    return window_list, label_list


def DTDP_input_from_multiple_samples(samplelist: List[Sample], width=100):
    """Makes all windowed inputs for multiple epgs"""
    windows_total = []
    labels_total = []
    for sample in samplelist:
        if len(sample.name) == 3:
            windows, labels = create_DTDP_inputs_from_sample(sample, width=width)
            windows_total.extend(windows)
            labels_total.extend(labels)
    inputs = DTDPTrainInput(samplelist, np.array(windows_total), np.array(labels_total))
    return inputs


def input_from_multiple_samples(samplelist: List[Sample], width: int, leftoffset: int, cutoff: int, normalised=True):
    """For a set of electropherograms, creates all input (node) images and their labels."""
    all_data = []
    all_data_normalised = []
    all_labels = []
    sample_names = []
    for sample in samplelist:
        if len(sample.name) == 3:
            sample_names.append(str(sample.name) + "." + str(sample.replica))
            sample_data = sample.data[leftoffset:cutoff, :width]
            all_data.append(sample_data)
            new = sample_data - np.min(sample_data)
            normalised_data = new / np.max(new)
            all_data_normalised.append(normalised_data)
            labels = find_peaks_flowing_out_of_bins(sample.data, sample.name)
            all_labels.append(labels[leftoffset:cutoff, :width])
    if normalised:
        input_from_samples = TrainInput(np.array(all_data_normalised), np.array(all_labels))
    else:
        input_from_samples = TrainInput(np.array(all_data), np.array(all_labels))
    return all_data, input_from_samples, sample_names


def input_from_multiple_PROVEDIt_samples(samplelist: List[Sample], width: int, leftoffset: int, rightcutoff: int,
                                         normalised=True):
    """For a set of PROVEDIt samples, creates all input images and their labels."""
    all_data = []
    all_data_normalised = []
    all_labels = []
    sample_names = []
    for sample in samplelist:
        sample_names.append(str(sample.name))
        sample_data = sample.data[leftoffset:rightcutoff, :width]
        all_data.append(sample_data)
        new = sample_data - np.min(sample_data)
        normalised_data = new / np.max(new)
        all_data_normalised.append(normalised_data)
        labels = np.zeros(9000)  # find_peaks_flowing_out_of_bins(sample.data, sample.name, alt = True)
        all_labels.append(labels[leftoffset:rightcutoff])
    if normalised:
        input_from_samples = TrainInput(np.array(all_data_normalised), np.array(all_labels))
    else:
        input_from_samples = TrainInput(np.array(all_data), np.array(all_labels))
    return all_data, input_from_samples, sample_names


def OLD_find_peaks_in_bins(sampledata, samplename):
    """Makes array of True/False of same size as data. True if a peak should theoretically be visible\
    based on the composition and the bin locations, False otherwise."""
    # Returns a nx6 numpy array
    person_mix = make_person_mixture(samplename)
    peaks = person_mix.create_peaks()
    bin_indices = [[], [], [], [], [], []]
    # cannot find precise index, since "size" of bins is accurate to 2 decimals, measurements to 1
    for peak in peaks:
        left_index = round((peak.allele.mid - peak.allele.left) * 10)
        right_index = round((peak.allele.mid + peak.allele.right) * 10)
        dye_index = peak.allele.dye.plot_index - 1
        # intervals are all about 1 nucleotide wide at most, 0.8 at least
        bin_indices[dye_index] += list(np.arange(left_index, right_index))

    indices = []
    for dye_index in range(6):
        bin_data = bin_indices[dye_index]
        sample_data = sampledata[:, dye_index]
        new_indices = [True if ind in bin_data else False for ind in range(len(sample_data))]
        indices.append(new_indices)

    return np.array(indices).transpose()


def find_peaks_flowing_out_of_bins(sampledata, samplename, alt=False):
    """Makes array of same size as data with True/False values if peak should be visible.
    Uses maximum within bin and follows whatever part of peak is visible to entire peak."""
    bin_edges = [[], [], [], [], [], []]
    if not alt:
        person_mix = make_person_mixture(samplename)
        peaks = person_mix.create_peaks()
    else:
        person_mix = OLD_make_person_mix_PROVEDIt(samplename)
        peaks = person_mix.create_peaks_no_heights()
    # cannot find precise index, since "size" of bins is accurate to 2 decimals, measurements to 1
    for peak in peaks:
        left_index = round((peak.allele.mid - peak.allele.left) * 10)
        right_index = round((peak.allele.mid + peak.allele.right) * 10)
        dye_index = peak.allele.dye.plot_index - 1
        # intervals are all about 1 nucleotide wide at most, 0.8 at least
        bin_edges[dye_index].append((left_index, right_index))
    if not alt:
        for size_std_peak in [600, 650, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2250, 2500, 2750, 3000, 3250, 3500,
                              3750, 4000, 4250, 4500, 4750, 5000]:
            bin_edges[5].append((size_std_peak - 4, size_std_peak + 4))
    else:
        for size_std_peak in [600, 800, 1000, 1140, 1200, 1400, 1600, 1800, 2000, 2140, 2200, 2400, 2500, 2600, 2800,
                              3000, 3140, 3200, 3400, 3600, 3800, 4000, 4140, 4200, 4400, 4600, 4800, 5000, 5140, 5200]:
            bin_edges[5].append((size_std_peak - 4, size_std_peak + 4))

    indices = []
    for color in range(6):
        bin_data = bin_edges[color]
        sample_data = sampledata[:, color]
        bin_booleans = [False] * len(sample_data)
        for left, right in bin_data:
            max_index = left + np.argmax(sample_data[left:right + 1])
            if sample_data[max_index - 1] > sample_data[max_index]:
                while sample_data[max_index - 1] - sample_data[max_index] > 1e-10:
                    max_index -= 1
            elif sample_data[max_index + 1] > sample_data[max_index]:
                while sample_data[max_index + 1] - sample_data[max_index] > 1e-10:
                    max_index += 1
            bin_booleans[max_index] = True
            left_end = max_index
            right_end = max_index
            while sample_data[left_end] - sample_data[left_end - 1] > 1e-10:
                left_end -= 1
                bin_booleans[left_end] = True
            while sample_data[right_end] - sample_data[right_end + 1] > 1e-10:
                right_end += 1
                bin_booleans[right_end] = True
        indices.append(bin_booleans)
    # array returned has same dimensions as sample array
    return np.array(indices).transpose()
