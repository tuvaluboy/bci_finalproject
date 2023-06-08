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
Applying Power Spectral Density Analysis to EEG Data
Power Spectral Density (PSD) analysis is a powerful tool for understanding the frequency content of EEG signals, which are electrical recordings of brain activity.

Data Loading and Preprocessing: The first step is to load the EEG dataset, which contains time-series data from multiple channels. Ensuring the data is clean and preprocessed is crucial for accurate analysis.

Channel Selection: EEG data consists of recordings from various electrodes. For focused analysis, we select 14 standard channels - 'Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'P3', 'Pz', 'P4', 'O1', and 'O2'. These channels capture significant aspects of brain activity.

PSD Analysis: With the data prepared, PSD analysis is performed on each selected channel. This involves calculating how the power of the EEG signals is distributed across different frequencies. The analysis helps in identifying dominant frequency bands in brain oscillations.

Visualization: Finally, the results are visualized through plots where the power (usually in dB) is plotted against frequency (in Hz) for each channel. This provides insights into the characteristics of brain oscillations within different frequency bands like delta, theta, alpha, beta, and gamma.

Analyzing the PSD of EEG data is essential for investigating neural dynamics and can be employed in various applications such as cognitive neuroscience, brain-computer interfaces, and the study of neurological disorders.

