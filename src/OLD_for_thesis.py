# Some functions used to create boxplots for thesis
# will be deleted


import matplotlib.pyplot as plt
import data_prep_functions as dpf, reading_functions as rf, \
    plotting_6C_functions as pf6, results as r

tracedata = ['TraceDataSet11.txt', 'TraceDataSet12.txt', 'TraceDataSet21.txt', 'TraceDataSet22.txt',
             'TraceDataSet31.txt', 'TraceDataSet32.txt', 'TraceDataSet41.txt', 'TraceDataSet42.txt',
             'TraceDataSet51.txt', 'TraceDataSet52.txt', 'TraceDataSet61.txt', 'TraceDataSet62.txt']


# to speed up tests, only do first dataset


def make_dataframe():
    # first create a list of all samples
    samples = []
    for elt in tracedata:
        samples += rf.txt_read_sample(elt)
    leftoffset = 500
    cutoff = 4800 + 500
    number_of_dyes = 6
    original_sampledata, inputs_for_unet, sample_names = dpf.input_from_multiple_samples(samples, number_of_dyes,
                                                                                         leftoffset, cutoff, True)
    print(len(sample_names), sample_names)
    # unet_model = trf.unet(inputs_for_unet, rightcutoff - leftoffset, 'data/weights_norm_avgpool.h5', False)
    # donor_sets = []
    # mix_types = []
    # number_donors = []
    # F1_scores = []
    # F1_scores_corrected = []
    # F1_scores_analyst = []
    # new_sample_names = []
    # for sample_number in range(len(sample_names)):
    #     sample_name, replica = sample_names[sample_number].split(".")
    #     # sample_data = original_sampledata[sample_number]
    #     input_example = inputs_for_unet.data[sample_number,:,:].reshape(1,rightcutoff-leftoffset,number_of_dyes,1)
    #     # label_example = inputs_for_unet.labels[sample_number, :, :]
    #     output_example = unet_model.predict(input_example).reshape(4800,6)
    #     actual_peaks = ppf.list_all_peaks(sample_name)
    #     corrected_peaks, augmented_output = ppf.check_correct_alleles_first(actual_peaks, output_example, leftoffset, 15)
    #     restofpeaks = ppf.mult_peaks(augmented_output, 0.5, leftoffset)
    #     corrected_peaks.extend(restofpeaks)
    #     predicted_peaks = ppf.mult_peaks(output_example, 0.5, leftoffset)
    #     if sample_name != "3E2":
    #         analyst_peaks = rf.shallow_analyst(sample_name)[int(replica)-1]
    #         F1_scores_analyst.append(ppf.F1_score(actual_peaks, analyst_peaks))
    #         F1_scores.append(ppf.F1_score(actual_peaks, predicted_peaks))
    #         F1_scores_corrected.append(ppf.F1_score(actual_peaks, corrected_peaks))
    #         new_sample_names.append(sample_names[sample_number])
    #         donor_set, mix_type, number_donor = sample_name
    #         donor_sets.append(donor_set)
    #         mix_types.append(mix_type)
    #         number_donors.append(number_donor)
    # r.store_dataframe(new_sample_names, np.array([donor_sets, mix_types, number_donors, F1_scores, F1_scores_corrected, F1_scores_analyst]).transpose())


def scores_only():
    df = r.load_dataframe()
    # print(df.describe())
    # print(df[df['upper'] == df['upper'].min()])
    r.make_boxplot(df, ['analyst', 'score'], 'donors')


def last_plots():
    # first create a list of all samples
    samples = []
    for elt in tracedata:
        samples += rf.txt_read_sample(elt)
    leftoffset = 500
    cutoff = 4800 + 500
    number_of_dyes = 6
    original_sampledata, inputs_for_unet, sample_names = dpf.input_from_multiple_samples(samples, number_of_dyes,
                                                                                         leftoffset, cutoff, True)
    # unet_model = trf.unet_train_test_split(inputs_for_unet, cutoff - leftoffset, 'data/weights_norm_avgpool.h5', False)

    new_sample_names = []
    for sample_number in range(1):  # len(sample_names)):
        sample_name, replica = sample_names[sample_number].split(".")
        sample_data = original_sampledata[sample_number]
        input_example = inputs_for_unet.data[sample_number, :, :].reshape(1, cutoff - leftoffset, number_of_dyes, 1)
        label_example = inputs_for_unet.labels[sample_number, :, :]
        output_example = unet_model.predict(input_example).reshape(4800, 6)
        if sample_name != "3E2":
            new_sample_names.append(sample_names[sample_number])
            positive_input = (sample_data[label_example == True]).flatten()
            positive_output = (output_example[label_example == True]).flatten()
            negative_input = (sample_data[label_example == False]).flatten()
            negative_output = (output_example[label_example == False]).flatten()
            plt.figure()
            plt.scatter(x=positive_input, y=positive_output, c="g", marker=".", alpha=0.2)
            plt.scatter(x=negative_input, y=negative_output, c="purple", marker=".", alpha=0.2)
            plt.show()
            pf6.plot_inputs_unet(sample_data, label_example)
