from datetime import datetime
import sys
import os.path
import numpy as np
sys.path.insert(0, "./src")

from src import data_prep_functions as dpf, reading_functions as rf, training_functions as trf

tracedata_for_training = ['TraceDataSet11.txt', 'TraceDataSet12.txt', 'TraceDataSet21.txt', 'TraceDataSet22.txt',
             'TraceDataSet31.txt', 'TraceDataSet32.txt', 'TraceDataSet41.txt', 'TraceDataSet42.txt',
             'TraceDataSet51.txt', 'TraceDataSet52.txt']
tracedata_for_testing = ['TraceDataSet61.txt', 'TraceDataSet62.txt']
#
# PROVEDIt_sized_trace_data_mix = ['PROVEDIt_RD14-0003(021016ADG_15sec)_sized_improved1.txt', 'PROVEDIt_RD14-0003(021016ADG_15sec)_sized_improved2.txt']
# PROVEDIt_sized_trace_data_SS = ['PROVEDIt_RD14-0003(100115ADG_15sec)_sized_improved1.txt', 'PROVEDIt_RD14-0003(100115ADG_15sec)_sized_improved2.txt']
#
#
# PROVEDIt_raw_trace_data_mix = ['PROVEDIt_RD14-0003(021016ADG_15sec)_raw_improved1.txt', 'PROVEDIt_RD14-0003(021016ADG_15sec)_raw_improved2.txt']
# PROVEDIt_raw_trace_data_SS = ['PROVEDIt_RD14-0003(100115ADG_15sec)_raw_improved1.txt', 'PROVEDIt_RD14-0003(100115ADG_15sec)_raw_improved2.txt']


if __name__ == '__main__':

    # my_model = unet_small()
    # my_model.summary()
    # new_model = MHCNN_DT()
    # new_model.summary()

    number_of_dyes = 6
    leftoffset = 500
    rightcutoff = 4800 + 500

    train_samples = []
    for elt in tracedata_for_training:
        train_samples += rf.txt_read_sample(elt)

    test_samples = []
    for elt in tracedata_for_testing:
        test_samples += rf.txt_read_sample(elt)

    unnormalised_train, train_input, names_train = dpf.input_from_multiple_samples(train_samples, number_of_dyes, leftoffset, rightcutoff, True)
    unnormalised_test, test_input, names_test = dpf.input_from_multiple_samples(test_samples, number_of_dyes, leftoffset, rightcutoff, True)
    Unet = trf.unet_train_test_split(train_input, test_input, 4800, "data/weights_NFI/weights_clocktime.h5", train=False,epochs=100)

    width = 80
    input_dim = 6*(width*2+1)
    DTDP_train_input = dpf.DTDP_input_from_multiple_samples(train_samples, width = width)
    DTDP_test_input = dpf.DTDP_input_from_multiple_samples(test_samples, width = width)
    start_time_training = datetime.now()
    FFN_model, metrics = trf.FFN(DTDP_train_input, DTDP_test_input, weightpath="data_for_github/weights_FFN_400.h5", inputsize=(input_dim,), train = False, epochs=400, batchsize=100)
    end_time_training = datetime.now()
    print('Total training time (400 epochs): ', end_time_training-start_time_training)

    #just_the_data = DTDP_test_input.data
    # start_time_training = datetime.now()
    # for ind in range(4800):
    #     FFN_model.predict(just_the_data[ind].reshape(1,966))
    # end_time_training = datetime.now()
    # print('predicting single epg FFN: ', end_time_training-start_time_training)

    # images, prediction = ppf.combine_results_FFN(DTDP_test_input, test_input, FFN_model, input_dim)
    # for index_of_image in range(DTDP_test_input.data.shape[0]):
    #     input = DTDP_test_input.data[index_of_image]
    #     pf.plot_results_FFN(input[:,0], prediction[:,:,0], test_input.labels[index_of_image,:,0], "Unet_fully_trained_on_"+names_test[index_of_image])
    # for sample_index in range(len(names_test)):
    #     sample = test_input.data[sample_index]
    #     truth = test_input.labels[sample_index, :, 0]
    #     result = Unet.predict(sample.reshape(1,4800,6))[:,:,0].reshape(4800)
    #     pf.plot_results_FFN(sample[:,0].reshape(4800), result, truth)
    #     auc = AUC()
    #     auc.update_state(truth, result)
    #     binacc = BinaryAccuracy()
    #     binacc.update_state(truth, result)
    # dataframe = pd.read_csv('data_for_github/FFNvUnet_scores.csv')
    # pf.scatterplot_scores_noc(dataframe)
    # pf.scatterplot_scores_mix(dataframe)
