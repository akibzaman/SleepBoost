from eeglib.helpers import EDFHelper
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import mne
import copy
from mne.time_frequency import psd_welch
import pandas as pd
from sklearn.neighbors import KDTree
from numba import njit

allFilePath = glob('prepare_datasets/data_edf_20/*PSG.edf')
allFilePath.sort()
labelFilePath = glob('prepare_datasets/data_edf_20/*Hypnogram.edf')
labelFilePath.sort()

###All Feature Extraction Function
def eegPowerBand(epochs):
    """EEG relative power band feature extraction.
    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    # FREQ_BANDS = {"delta": [0.5, 4.5],
    #               "theta": [4.5, 8.5],
    #               "alpha": [8.5, 11.5],
    #               "sigma": [11.5, 15.5],
    #               "beta": [15.5, 30]
    #               }
    FREQ_BANDS = {"lowdelta": [0.5, 2.0],
                  "highdelta": [1.2, 4],
                  "theta": [4, 8],
                  "alpha": [8, 13],
                  # "sigma": [11.5, 15.5],
                  "lowbeta": [13, 20],
                  "highbeta": [20, 30],
                  "lowgama": [30, 45]
                  }
    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=45)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))
    return np.concatenate(X, axis=1)

def totalPowerBand(data):
    X = copy.deepcopy(data)
    return np.concatenate((data,np.vstack(np.sum(X,axis=1))),axis=-1)

def derivedPowerFeatures(data):
    D1 = np.vstack(np.divide(data[:,0],data[:,-1]))
    D2 = np.vstack(np.divide(data[:,1],data[:,-1]))
    D3 = np.vstack(np.divide(data[:,2],data[:,-1]))
    D4 = np.vstack(np.divide(data[:,3],data[:,-1]))
    D5 = np.vstack(np.divide(data[:,4]+ data[:,5],data[:,-1]))
    D6 = np.vstack(np.divide(data[:,6],data[:,-1]))
    D7 = np.vstack(np.divide(data[:,2]+ data[:,3],data[:,-1]))
    D8 = np.vstack(np.divide(data[:,3],data[:,4]+data[:,5])) # E5 /(E6 + E7 )
    D9 = np.vstack(np.divide(data[:,2]+data[:,3],data[:,3]+data[:,4]+data[:,5])) #(E 4 + E 5 )/(E 5 + E 6 + E 7 )
    D10 = np.vstack(np.divide(data[:,2],data[:,4]+data[:,5])) #E 4 /(E 6 + E 7 )
    D11 = np.vstack(np.divide(data[:,1],data[:,2]+data[:,3])) #E 3 /(E 4 + E 5 )
    D12 = np.vstack(np.divide(data[:,2],data[:,1]+data[:,3])) #E 4 /(E 3 + E 5 )
    D13 = np.vstack(np.divide(data[:,3],data[:,1]+data[:,2])) #E 5 /(E 3 + E 4 )
    D14 = np.vstack(np.divide(data[:,0],data[:,1]+data[:,-1])) #E 2 /(E 3 + E 9 )
    D15 = np.vstack(np.divide(data[:,3],data[:,-1])) #E 5 /E 9
    D16 = np.vstack(np.divide(data[:,4]+ data[:,5],data[:,-1])) #(E 6 + E 7 )/E 9
    D17 = np.vstack(np.divide(data[:,3],data[:,2]))#E 5 /E 4
    D18 = np.vstack(np.divide(data[:,0]+ data[:,1],data[:,2]))#E 2 + E 3 )/E 4
    D19 = np.vstack(np.divide(data[:,0]+ data[:,4],data[:,-1]))#E 2 + E 6 )/E 9
    return np.concatenate((D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,D16,D17,D18,D19),axis=-1)

def mean(x):
    return np.vstack(np.mean(x, axis=-1))
def std(x):
    return  np.vstack(np.std(x,axis=-1))
def var(x):
    return np.vstack(np.var(x,axis=-1))
def minim(x):
    return np.vstack(np.min(x,axis=-1))
def maxim(x):
    return np.vstack(np.max(x,axis=-1))
def argminim(x):
    return np.vstack(np.argmin(x,axis=-1))
def argmaxim(x):
    return np.vstack(np.argmax(x,axis=-1))
def rms(x):
    return np.vstack(np.sqrt(np.mean(x**2,axis=-1)))
def absDiffSignal(x):
    return np.vstack(np.sum(np.abs(np.diff(x,axis=-1)),axis=-1))
def skewness(x):
    return np.vstack(stats.skew(x,axis=-1))
def kurtosis(x):
    return np.vstack(stats.kurtosis(x,axis=-1))
def median(x):
    return np.vstack(np.median(x,axis=-1))
def hjorthMobility(data):
    return np.vstack(np.sqrt(np.var(np.gradient(data,axis=-1),axis=-1) / np.var(data,axis=-1)))
def hjorthComplexity(data):
    return np.vstack(hjorthMobility(np.gradient(data,axis=-1)) / hjorthMobility(data))
def allTimeDomainFeatures(x):
    return np.concatenate((mean(x),std(x),var(x),minim(x),maxim(x),argminim(x),
                           argmaxim(x),rms(x), median(x), absDiffSignal(x),skewness(x),kurtosis(x)
                           ,hjorthMobility(x),hjorthComplexity(x)),axis=-1)

# rawData = mne.io.read_raw_edf(allFilePath[0],preload=True)
# rawData.set_eeg_reference()
# rawData.pick_channels(['EEG Fpz-Cz'])
# epochs = mne.make_fixed_length_epochs(rawData,duration=30,overlap=1)
# print(epochs.get_data().shape)
# tfr = mne.time_frequency.tfr_morlet(epochs,n_cycles=3,return_itc=False)
# print(tfr.shape)
for i in range(len(allFilePath)):
    raw_train = mne.io.read_raw_edf(allFilePath[i], preload=True)
    raw_train = raw_train.pick_channels(['EEG Fpz-Cz'])
    annot_train = mne.read_annotations(labelFilePath[i])
    raw_train.set_annotations(annot_train, emit_warning=False)

    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}

    # keep last 30-min wake events before sleep and first 30-min wake events after
    # sleep and redefine annotations on raw data
    annot_train.crop(annot_train[1]['onset'] - 30 * 60,
                     annot_train[-2]['onset'] + 30 * 60)
    raw_train.set_annotations(annot_train, emit_warning=False)

    events_train, _ = mne.events_from_annotations(
        raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    # create a new event_id that unifies stages 3 and 4
    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3/4': 4,
                'Sleep stage R': 5}
    tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included

    epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                              event_id=event_id, tmin=0., tmax=tmax, baseline=None)

    labels = events_train[: , 2]
    time_features= allTimeDomainFeatures(epochs_train.get_data())
    # print(epochs_train)
    M = eegPowerBand(epochs_train)
    E1 = totalPowerBand(M)
    D1 = derivedPowerFeatures(E1)
    F1 = np.concatenate((time_features,E1,D1),axis=1)
    DATA = pd.DataFrame(F1, columns = ['mean' ,'std' ,'var' ,'minim' ,'maxim' ,'argminim' , 'argmaxim' ,
                                       'rms' ,'median','absDiffSignal' ,'skewness' ,'kurtosis' , 'HM' , 'HC',
                                       'E2','E3','E4', 'E5', 'E6', 'E7','E8', 'E1',
                                       'D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13',
                                       'D14','D15','D16','D17','D18','D19'])
    LABELS = pd.DataFrame(labels, columns=['class'])
    FINALDATA = pd.concat((DATA, LABELS),axis=1)
    FINALDATA.to_csv("alldatacsv/sample"+str(i)+".csv", index=None)