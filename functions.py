# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:32:27 2023

@author: danie
"""

import mne
import numpy as np
import numpy
import scipy 
import pandas as pd 

def get_features (x):
    #reading edf files 
    raw = mne.io.read_raw_edf (x, preload=True, stim_channel='auto', verbose=False)
    fs = 256
    
    #bandwidth filtering 
    f_l=0.5
    f_h=60.0
    raw.filter(f_l, f_h, fir_design='firwin', skip_by_annotation='edge')
   
    #notch filtering
    raw.notch_filter(50)
 
      #no need in final thesis
   # ica = mne.preprocessing.ICA(n_components =15, random_state=97, max_iter=800)
   # ica.fit(raw)
   # raw.load_data()
   # ica.exclude = [0,1,2,3]
   # ica.apply(raw)
    
    # function for normalization of the signal 
    def normalize(eeg, level):
        amp_new = 10**(level / 20)
        amp_max = np.max(np.abs(eeg))
        return amp_new * eeg / amp_max
    #function for bandpower calculaiton 
    
    data = raw.get_data()
    channels = raw.ch_names
    data = data.transpose()
    df = pd.DataFrame(data = data, columns = channels)
    F3 = normalize(df['EEG F3-LE'].values, 0)
    F4 = normalize(df['EEG F4-LE'].values, 0)
  
    
    def bandpower(x, fs, fmin, fmax):
        f, Pxx = scipy.signal.welch(x, fs=fs,window='hann', nperseg=4*fs)
        ind_min = numpy.argmax(f > fmin) - 1
        ind_max = numpy.argmax(f > fmax) - 1
        return numpy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
    
    def totalpower(x, fs):
        f, Pxx = scipy.signal.welch(x, fs=fs,window='hann', nperseg=4*fs)
        return numpy.trapz(Pxx, f)
    
    def spectral_entropy (x, fs):
        _, psd = scipy.signal.welch(x, fs=fs,window='hann', nperseg=4*fs)
        psd_norm = np.divide(psd, psd.sum())
        se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
        se /= np.log2(psd_norm.size)
        return se 
    
    def mean_psd (x, fs):
        _,  psd = scipy.signal.welch(x, fs=fs,window='hann', nperseg=4*fs)
        mps = np.mean(psd)
        return mps

    out = pd.DataFrame()
    n_win = 30*fs
    n_hop = n_win//2
    pos = 0 
    while  (pos<=F3.size-n_win):
        
        frameF3= F3[pos:pos+n_win]
        frameF4= F4[pos:pos+n_win]
        
        F3_pow_delta_abs = bandpower(frameF3, fs, 0.5, 4)
        F4_pow_delta_abs = bandpower(frameF4, fs, 0.5, 4)
        
        F3_pow_teta_abs = bandpower(frameF3, fs, 4, 8)
        F4_pow_teta_abs = bandpower(frameF4, fs, 4, 8)
        
        F3_pow_alfa_abs = bandpower(frameF3, fs, 8, 13)
        F4_pow_alfa_abs = bandpower(frameF4, fs, 8, 13)
        
        F3_pow_beta_abs = bandpower(frameF3, fs, 13, 32)
        F4_pow_beta_abs = bandpower(frameF4, fs, 13, 32)
        
        F3_pow_gama_abs = bandpower(frameF3, fs, 32, 60)
        F4_pow_gama_abs = bandpower(frameF4, fs, 32, 60)
        
        F3_pow_tot = totalpower(frameF3, fs)
        F4_pow_tot = totalpower(frameF4, fs)
        
        F3_pow_delta_rel = F3_pow_delta_abs/ F3_pow_tot 
        F4_pow_delta_rel =  F4_pow_delta_abs/F4_pow_tot 
        
        F3_pow_teta_rel = F3_pow_teta_abs/ F3_pow_tot
        F4_pow_teta_rel = F4_pow_teta_abs/ F4_pow_tot
        
        F3_pow_alfa_rel = F3_pow_alfa_abs/ F3_pow_tot
        F4_pow_alfa_rel = F4_pow_alfa_abs/ F4_pow_tot
        
        F3_pow_beta_rel = F3_pow_beta_abs/ F3_pow_tot
        F4_pow_beta_rel = F4_pow_beta_abs/ F4_pow_tot
        
        F3_pow_gama_rel = F3_pow_gama_abs/ F3_pow_tot
        F4_pow_gama_rel = F4_pow_gama_abs/ F4_pow_tot
        
        se_F3 = spectral_entropy(frameF3, fs)
        se_F4 = spectral_entropy(frameF4, fs)
        
        mpsd_F3 = mean_psd (frameF3, fs)
        mpsd_F4 = mean_psd (frameF4, fs)
        
        alpha = np.log((F4_pow_alfa_abs/F3_pow_alfa_abs))
        
        pom = pd.DataFrame([F3_pow_delta_abs, F4_pow_delta_abs, 
                  F3_pow_teta_abs,  F4_pow_teta_abs, 
                  F3_pow_alfa_abs, F4_pow_alfa_abs,
                  F3_pow_beta_abs, F4_pow_beta_abs, 
                  F3_pow_gama_abs,  F4_pow_gama_abs, 
                  F3_pow_tot,  F4_pow_tot, 
                  F3_pow_delta_rel, F4_pow_delta_rel, 
                  F3_pow_teta_rel,  F4_pow_teta_rel, 
                  F3_pow_alfa_rel, F4_pow_alfa_rel,
                  F3_pow_beta_rel, F4_pow_beta_rel, 
                  F3_pow_gama_rel,  F4_pow_gama_rel,
                  mpsd_F3, mpsd_F4,
                  se_F3, se_F4,
                  alpha])
        
        pom = pom.T
        out = pd.concat([out,pom], ignore_index=True) 
        pos = pos + n_hop

    return out

def depression_predict (df, pom):
    df=pd.concat([df,pom])
    df=df.T
    df.columns=(['EC', 'EO', 'TASK',
                 'F3_pow_delta_abs', 'F4_pow_delta_abs', 
                  'F3_pow_teta_abs', 'F4_pow_teta_abs', 
                  'F3_pow_alfa_abs', 'F4_pow_alfa_abs',
                  'F3_pow_beta_abs', 'F4_pow_beta_abs', 
                  'F3_pow_gama_abs', ' F4_pow_gama_abs', 
                  'F3_pow_tot',  'F4_pow_tot', 
                  'F3_pow_delta_rel', 'F4_pow_delta_rel', 
                  'F3_pow_teta_rel',  'F4_pow_teta_rel', 
                  'F3_pow_alfa_rel', 'F4_pow_alfa_rel',
                  'F3_pow_beta_rel', 'F4_pow_beta_rel', 
                  'F3_pow_gama_rel',  'F4_pow_gama_rel',
                  'mpsd_F3', 'mpsd_F4',
                  'se_F3', 'se_F4',
                  'alpha'])
    return df

