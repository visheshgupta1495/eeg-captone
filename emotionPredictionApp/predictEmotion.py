import os
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from keras.models import load_model
import pyeeg as pe
import scipy.signal as sg
from sklearn.preprocessing import normalize
import pandas as pd
from keras.models import Model
import warnings
warnings.filterwarnings('ignore')

# Get the current working directory
# cwd = os.path.dirname(os.getcwd())
cwd = os.getcwd()

channel = [1, 2, 3, 4, 7, 11, 13, 17, 19, 20, 21, 25, 29, 31]
band = [32, 33, 34, 36, 38, 39, 40, 42, 44, 46, 48, 50]
window_size = 255
step_size = 16
sample_rate = 128
subjectList = ['01', '02', '03', '04', '05', '06', '07',
               '08', '09', ] + [str(x) for x in range(10, 33)]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sg.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_2(lowcut, highcut, fs, order=5):
    nyq = 1 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = sg.butter(order, [low, high], analog=False,
                    btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass_2(lowcut, highcut, fs, order=order)
    y = sg.sosfilt(sos, data)
    return y


def FFT_Processing(x, band, window_size, step_size, sample_rate):
    '''
    arguments:  string subject
                list channel indice
                list band
                int window size for FFT
                int step size for FFT
                int sample rate for FFT
    return:     void
    '''
    meta = []
    data = []

    for i in range(14):
        data.append(butter_bandpass_filter(x[i], 4, 45, 128))
    data = np.array(data)
    print(np.shape(data))

    start = 0

    fs = 228
    lowcut = 25.0
    highcut = 55.0
    b, a = butter_bandpass(lowcut, highcut, fs,  order=3)

    while start + window_size < data.shape[1]:
        meta_array = []
        meta_data = []  # meta vector for analysis
        for j in range(14):
            # Slice raw data over 2 sec, at interval of 0.125 sec
            X = data[j][start: start + window_size]
            Y_fir = sg.lfilter(b, a, X)
            # FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
            Y = pe.bin_power(Y_fir, band, sample_rate)
            meta_data = meta_data + list(Y[0])
            # meta_data = meta_data + list(Y_fir)
        # print(np.shape(labels), np.shape(meta_data), np.shape(Y[0]),np.shape(X), np.shape(Y_fir))
        meta_array.append(np.array(meta_data))
        meta.append(np.array(meta_array))
        start = start + step_size

    meta = np.array(meta)
    return meta
    # np.save('/content/drive/MyDrive/Deap Dataset/emotiv/data_preprocessed/s' + sub, meta, allow_pickle=True, fix_imports=True)
    print("Done")


def emotionPred(file):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # class_dict = {0: 'diseased cotton leaf',
    #               1: 'diseased cotton plant',
    #               2: 'fresh cotton leaf',
    #               3: 'fresh cotton plant'}
    class_dict = {0: 'Cotton leaf is Diseases.',
                  1: 'Cotton plant is Diseased.',
                  2: 'Its a fresh Cotton leaf',
                  3: 'Its a fresh Cotton plant'}

    print(cwd)
    df = pd.read_csv(cwd + file)

    df = df[[
        "EEG.AF3",
        "EEG.F3",
        "EEG.F7",
        "EEG.FC5",
        "EEG.T7",
        "EEG.P7",
        "EEG.O1",
        "EEG.AF4",
        "EEG.F4",
        "EEG.F8",
        "EEG.FC6",
        "EEG.T8",
        "EEG.P8",
        "EEG.O2"]]

    arr = df.to_numpy().transpose()
    print(np.shape(arr))

    result = FFT_Processing(arr, band, window_size, step_size, sample_rate)

    data_testing = []

    for i in range(0, result.shape[0]):
        data_testing.append(result[i][0])
    X = data_testing
    X = normalize(X)
    x_test = np.array(X[:])

    scaler = RobustScaler()
    x_test = scaler.fit_transform(x_test)

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    model_ar = load_model(
        cwd+"/media/Emotive fft_fir_cnn_80_20_model_arousal valacc 0.8749.h5")
    model_val = load_model(
        cwd+"/media/Emotive fft_fir_cnn_80_20_model_valence valacc 0.8696.h5")
    model_dom = load_model(
        cwd+"/media/Emotive fft_fir_cnn_80_20_model_domain valacc 0.8787.h5")
    model_lik = load_model(
        cwd+"/media/Emotive fft_fir_cnn_80_20_model_liking valacc 0.8695.h5")

    y_pred_ar = model_ar.predict(x_test)
    y_pred_val = model_val.predict(x_test)
    y_pred_dom = model_dom.predict(x_test)
    y_pred_lik = model_lik.predict(x_test)

    y_classes_ar = [np.argmax(y, axis=None, out=None) for y in y_pred_ar]
    y_classes_val = [np.argmax(y, axis=None, out=None) for y in y_pred_val]
    y_classes_dom = [np.argmax(y, axis=None, out=None) for y in y_pred_dom]
    y_classes_lik = [np.argmax(y, axis=None, out=None) for y in y_pred_lik]

    ar_average = np.average(y_classes_ar)
    val_average = np.average(y_classes_val)
    dom_average = np.average(y_classes_dom)
    lik_average = np.average(y_classes_lik)

    output = {0: "Patient is Angry",
              1: "Patient is Happy", 2: "Patient is Sad"}

    output_lst = []
    output_lst.append(np.abs(3.1-val_average) + np.abs(3.5 -
                                                       ar_average) + np.abs(5.6-dom_average))
    output_lst.append(np.abs(6.1-val_average) + np.abs(5.5 -
                                                       ar_average) + np.abs(5.6-dom_average))
    output_lst.append(np.abs(2.8-val_average) + np.abs(4.4 -
                                                       ar_average) + np.abs(5.6-dom_average))
    pred_class = output[np.argmin(output_lst)]

    print(output_lst, pred_class)

    return (pred_class)
