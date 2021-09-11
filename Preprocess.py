import numpy as np
import pickle
from tqdm import tqdm
import os
from scipy.io import loadmat
import pandas as pd
from biosppy.signals import ecg as ecgprocess
import matplotlib.pyplot as plt
import cv2
import time
import gc

gc.enable()

def SaveAsPickle(varables,file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(varables, f)

def LoadPickle(file_name):
    file = pickle.load(open(file_name, 'rb'))
    return file
    


def ImportFile(path):
    
    file_list = []
      
    #root, directories, files
    for r, d, f in os.walk(path):        
        for file in f:
            if '.mat' in file:
                         
                file_dir = os.path.join(r, file)
                file_list.append(file_dir)
                
    file_list = sorted(file_list) #sorted list by file's name (eg. A00001 to A8528)
    
    signals = []
    for file in tqdm(file_list):
        sig = list(loadmat(file).values())[0][0]/1000
        signals.append(sig)
     
    #import reference 
    refer_path = os.path.join(path, 'REFERENCE.csv')
    reference = np.array(pd.read_csv(refer_path, header = None))
    label = reference[:,1]
    label[label == 'N'] = 0  #Normal
    label[label == 'A'] = 1  #Afib
    label[label == 'O'] = 2  #Other
    label[label == '~'] = 3  #Noise
    
    dataset = list(zip(label, signals))
           
    return dataset

def WindowSelection(signals, win_size= 3000, method= 'center', StartPoint = None):
    
    print('select window...')
    
    if method == 'center':
        Signals = []
        
        for sig in tqdm(signals):
            
            sig_len = len(sig)
            if sig_len < win_size:
                pad_num = win_size - sig_len
                pad_left = int(np.ceil(pad_num/2))
                pad_right = int(np.floor(pad_num/2))
                select_signal_win = np.pad(sig, (pad_left, pad_right), mode = 'constant')
                Signals.append(select_signal_win)
            else:
                
                start_point = int(np.round((sig_len - win_size)/2))
                end_point = start_point + win_size
                select_signal_win = sig[start_point:end_point]
                Signals.append(select_signal_win)
        
        return Signals
            
    elif method == 'random':
        Signals = []
        
        for sig in tqdm(signals):
            
            sig_len = len(sig)
            if sig_len < win_size:
                pad_num = win_size - sig_len
                pad_left = int(np.ceil(pad_num/2))
                pad_right = int(np.floor(pad_num/2))
                select_signal_win = np.pad(sig, (pad_left, pad_right), mode = 'constant')
                Signals.append(select_signal_win)
            else:
                max_start_point = sig_len - win_size
                start_point = np.random.randint(0, max_start_point)
                end_point = start_point + win_size
                select_signal_win = sig[start_point:end_point]
        return Signals
    
    elif method == 'fix':
        Signals = []
        
        for sig in tqdm(signals):
            
            sig_len = len(sig)
            if sig_len < win_size:
                pad_num = win_size - sig_len
                pad_left = int(np.ceil(pad_num/2))
                pad_right = int(np.floor(pad_num/2))
                select_signal_win = np.pad(sig, (pad_left, pad_right), mode = 'constant')
                Signals.append(select_signal_win)
            else:
                start_point = StartPoint
                end_point = start_point + win_size
                select_signal_win = sig[start_point:end_point]
                Signals.append(select_signal_win)
        
        return Signals
    
    else:
        print('Error: Please select method.')

def FeatureExtraction(dataset):
    Ts            = [] #Signal time axis reference (seconds).
    Filtered_ecg  = [] #Filtered ECG signal.
    Rpeaks        = [] #R-peak location indices.
    Templates_ts  = [] #Templates time axis reference (seconds).
    Templates     = [] #Extracted heartbeat templates.
    Heart_rate_ts = [] #Heart rate time axis reference (seconds).
    Heart_rate    = [] #Instantaneous heart rate (bpm).
    Label         = []
    for lb, sig in tqdm(dataset):
        ts, filt_ecg, rp, temp_ts, temp, hr_ts, hr = ecgprocess.ecg(sig, 300, False)
        
        Ts.append(ts)
        Filtered_ecg.append(filt_ecg)
        Rpeaks.append(rp)
        Templates_ts.append(temp_ts)
        Templates.append(temp)
        Heart_rate_ts.append(hr_ts)
        Heart_rate.append(hr)
        Label.append(lb)
        
    return Ts, Filtered_ecg, Rpeaks, Templates_ts, Templates, Heart_rate_ts, Heart_rate, Label 

#3000 iterations at one time, to avoid out of memory 
def PrepareTemplates(Templates, Templates_ts, save_path, start, end): #add start, end to avoid out of memory
    
    #check data match
    if len(Templates) != len(Templates_ts):
        raise ValueError
        
    for i in tqdm(range(start, end)):
        
        if i > len(Templates) - 1:
            #creat image file list
            file_list = []
            for j in range(len(Templates)):
                file = str(j)+'.png'
                file_list.append(file)    
            SaveAsPickle(file_list, os.path.join(save_path,'file_list.pk1'))
            break
        
        else:
            plt.plot(Templates_ts[i], Templates[i].T, 'm' ,alpha = 0.7)
            plt.axis('off')   #don't show axis
            plt.savefig(os.path.join(save_path, str(i)))
            plt.clf()
            plt.close('all')
                                                 

def FindMedianAmp(templates):
    Amplitude = []
    for temp in tqdm(templates):
        peak = []
        for amp in temp:
            peak.append(max(amp))
        med_val = np.percentile(peak, 50)
        
        select = (abs(peak - med_val)).argmin()
        
        Amplitude.append(temp[select])
        
    return Amplitude


def PrepareFeatures(train_path = './training2017', sample_path = './sample2017/validation'):
    
    #training dataset feature extraction
    training = ImportFile(train_path)
    if os.path.isdir('./feature') == False:
        os.mkdir('./feature') #creat folder to save features
    tn_Ts, tn_Filtered_ecg, tn_Rpeaks, tn_Templates_ts, tn_Templates, tn_Heart_rate_ts, tn_Heart_rate, tn_Label = FeatureExtraction(training)
    
    SaveAsPickle(tn_Ts, './feature/train_ts.pk1')
    SaveAsPickle(tn_Filtered_ecg, './feature/train_filtered_ecg.pk1')
    SaveAsPickle(tn_Rpeaks, './feature/train_Rpeak.pk1')
    SaveAsPickle(tn_Templates_ts, './feature/train_templates_ts.pk1')
    SaveAsPickle(tn_Templates, './feature/train_templates.pk1')
    SaveAsPickle(tn_Heart_rate_ts, './feature/train_HeartRate_ts.pk1')
    SaveAsPickle(tn_Heart_rate, './feature/train_HeartRate.pk1')
    SaveAsPickle(tn_Label, './feature/train_label.pk1')
    
    del tn_Ts, tn_Filtered_ecg, tn_Rpeaks, tn_Heart_rate_ts, tn_Heart_rate, tn_Label
    
    
    #validation dataset feature extraction
    sample = ImportFile(sample_path)
    
    tt_Ts, tt_Filtered_ecg, tt_Rpeaks, tt_Templates_ts, tt_Templates, tt_Heart_rate_ts, tt_Heart_rate, tt_Label = FeatureExtraction(sample)
    
    SaveAsPickle(tt_Ts, './feature/test_ts.pk1')
    SaveAsPickle(tt_Filtered_ecg, './feature/test_filtered_ecg.pk1')
    SaveAsPickle(tt_Rpeaks, './feature/test_Rpeak.pk1')
    SaveAsPickle(tt_Templates_ts, './feature/test_templates_ts.pk1')
    SaveAsPickle(tt_Templates, './feature/test_templates.pk1')
    SaveAsPickle(tt_Heart_rate_ts, './feature/test_HeartRate_ts.pk1')
    SaveAsPickle(tt_Heart_rate, './feature/test_HeartRate.pk1')
    SaveAsPickle(tt_Label, './feature/test_label.pk1')
    
    del tt_Ts, tt_Filtered_ecg, tt_Rpeaks, tt_Heart_rate_ts, tt_Heart_rate, tt_Label
    
    return tn_Templates, tn_Templates_ts, tt_Templates, tt_Templates_ts
 

def get_MedAmpInput():
     
    #tn_Templates, tn_Templates_ts, tt_Templates, tt_Templates_ts = PrepareFeatures()
    tn_Templates    = LoadPickle('./feature/train_templates.pk1')
    
    tt_Templates    = LoadPickle('./feature/test_templates.pk1')
    
    train_sig = FindMedianAmp(tn_Templates)
    test_sig  = FindMedianAmp(tt_Templates)
    train_lb  = LoadPickle('./feature/train_label.pk1')
    test_lb   = LoadPickle('./feature/test_label.pk1')
    
    train  = list(zip(train_lb, train_sig))
    test   = list(zip(test_lb, test_sig))
    SaveAsPickle(train, 'train_med_amp.pk1')
    SaveAsPickle(test, 'test_med_amp.pk1')

    
if __name__ == "__main__":
    _, _, _, _ = PrepareFeatures()
    get_MedAmpInput()


    
    


    
    
        
      
        
    
    
    
    

        
    

    

    


