# bci_finalproject
This work has been developed to fulfill the requisites of the course titled '11120ISA557300: Brain Computer Interfaces: Fundamentals and Application', under the guidance of Prof. Chun-Hsiang Chuang.

## Authors

- SYED ASIF AHMAD QADRI (110065859)
- FINI IUNI (111065425)
- JEAN CARLOS (111065422)


## Table of contents

- [Introduction](#introduction)
- [Demo Video](#demo-video)
- [Dataset](#dataset)
- [Model Framework](#model-framework)
- [Validation](#validation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Introduction
Emotion, a vital aspect of daily life, is predominantly conveyed through facial expressions, voice tones, and physiological signals like EEG and EOG. Among these, EEG signals are favored for emotion recognition due to their noninvasive nature and ease of use. Emotion recognition, crucial for regulating emotions and mental well-being, utilizes two models - the dimensional model and the discrete emotion model. The dimensional model is more intuitive and widely used as it represents emotions in a coordinate space with continuous values like arousal and valence. The advent of deep learning, particularly in EEG applications, has been transformative due to its capability to autonomously extract features and perform end-to-end classification.

Objectives:

- To recognize and regulate emotions, which are essential for mental health.
- To utilize physiological signals, especially EEG, for emotion recognition as they are not influenced by subjective factors.
- To employ the dimensional model for emotion recognition as it represents emotions using basic dimensions with continuous values.
- To leverage the capabilities of deep learning in EEG emotion recognition for automated feature extraction and classification.

## Demo Video


## Dataset

Dataset (SEED IV) 
SEED IV is an evolution of the original SEED dataset, provided by the Brain-like Computing & Machine Intelligence (BCMI) laboratory, which is led by Prof. Bao-Liang Lu. To gain access to the SEED IV dataset, an application is required. Please visit the BCMI laboratory’s website to apply for access before using this dataset.

Seventy-two film clips were carefully chosen through a preliminary study, which tended to include emotions such as <b>happiness</b>, <b>sadness</b>, <b>fear</b>, or <b>neutrality</b>. A total of 15 subjects participated in the experiment.

The experiment employed a 62-Channel ESI NeuroScan system and was designed as follows: each participant underwent three sessions on different days, with each session containing 24 trials. In each trial, the participant watched a film clip intended to induce a specific emotional state (happiness, sadness, fear, or a neutral state). While the participant was watching the clip, their EEG signals and eye movements were recorded.
<br>
<p align="center">
<img src="./imgs/raw dataset lcoations.jpg" alt="raw dataset" >
  </p>
<br>
 
The dataset comprises two files: eeg_raw_data and eye_raw_data. For the purpose of this experiment, we focused only on the eeg_raw_data. The eeg_raw_data folder contains the raw EEG signals from the 15 participants. Within eeg_raw_data, there are three folders named 1, 2, and 3, corresponding to the three sessions. Each .mat file is named in the format {subjectName}_{Date}.mat. These folders store a structure with fields named "cz_eeg1", "cz_eeg2", up to "cz_eeg24", which correspond to the EEG signal recorded during the 24 trials. For each of the signal processing, the raw EEG data are first downsampled to a 200 Hz sampling rate.
<br>
<p align="center">
<img src="./imgs/experiment framework.jpg" alt="experiment framework" >
  </p>
<br>

Our project, titled "Emotion Recognition from EEG Data Using Advanced Feature Extraction and Deep Learning Techniques", focuses on emotional responses. Therefore, we concentrated on 14 selected channels - the frontal (Fp1, Fp2, F3, F4), temporal (T3, T4), central (C3, Cz, C4), parietal (P3, Pz, P4), and occipital (O1, O2) for our analyses. 
<br>
<p align="center">
<img src="./imgs/14 channels.png" alt="14 channels" width="350px" height="auto"  >
 </p>
<br>
 Initially, time-frequency analysis was conducted on three EEG channels - FP1, FP2, and F3. This analysis involved decomposing the EEG signals into time and frequency domains, allowing for the examination of how the spectral characteristics of the signals change over time. This is particularly important for understanding the dynamics of brain activity.
<br>
<p align="center">
<img src="./imgs/time frequency analysis.png" alt="time frequency analysis" >
 </p>
<br>
 After the time-frequency analysis, PSD was applied to the same three channels (FP1, FP2, and F3) to analyze the power distribution over various frequency bands. Initially, a baseline removal was applied to the PSD to normalize the spectral power relative to a reference period. However, upon visual inspection, it was observed that the removal of the baseline did not make a significant difference in this specific case.
<br>
<p align="center">
  <table>
    <tr>
      <td>
        <img src="./imgs/psd.png" alt="psd" >
      </td>
      <td>
        <img src="./imgs/psd.png" alt="psd" >
      </td>
    </tr>
 </table>
     </p>
<br>
 Finally, ERSP was applied to the three channels to analyze how the spectral power changes in response to specific events. ERSP is used to determine the average changes in amplitude over time and frequency in response to a stimulus or event, as compared to a baseline period. This provides insight into event-related brain dynamics and helps in understanding how the brain processes specific stimuli.
<br>
<p align="center">
  <table>
    <tr>
      <td>
        <img src="./imgs/ERSP_FP1.png" alt="ERSP_FP1" >
      </td>
      <td>
        <img src="./imgs/ERSP_FP2.png" alt="ERSP_FP2" >
      </td>
      <td>
        <img src="./imgs/ERSP_F3.png" alt="ERSP_F3" >
      </td>
    </tr>
 </table>
     </p>
<br>
Overall, the combination of time-frequency analysis, PSD, and ERSP provided a comprehensive analysis of the EEG data for the three channels of interest. This suite of analyses can be invaluable in understanding both the temporal and spectral dynamics of brain activity, and in characterizing how the brain responds to specific events or stimuli.

