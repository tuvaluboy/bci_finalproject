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
- [Surveying and Analyzing existing literature](#surveying and analyzing existing literature)
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

## Surveying and Analyzing existing literature

The research studies on brain-computer interfaces (BCIs) and emotion recognition has grown a lot over the past decade. Many scientists and engineers have begin to explore the potential of BCIs in many application areas, including mental health and emotion regulation.

### 1. Brain Computer Interfaces (BCI)
BCIs have generated a lot of interest in recent years, as they promise to offer new ways to interact with environmental and technological systems directly through brain signals. BCI technology has been studied for a variety of applications, including assistive technology, gaming, and mental health (Wolpaw et al., 2002; Lotte et al., 2018).

### 2. Recognition and regulation of emotions.
Emotion recognition is a very complex cognitive process that allows us to identify and understand the emotions of others, which is essential for interpersonal communication and mental health (Kret & De Gelder, 2012). Studies have found that emotion regulation, which involves managing and responding to emotional experiences, is critical to mental health and well-being (Gross, 2014). Therefore, the recognition and regulation of emotions are vital in the search for better mental health interventions. Which can be of crucial help because mental issues on peoples due to emotions is a problem that has been on a rise in the recent years.

### 3. EEG for emotion recognition
EEG-based emotion detection has received increasing attention due to its potential to provide real-time objective indicators of emotional states. Scientists like Lin et al. (2010) and Koelstra et al. (2012) demonstrated the viability of using EEG signals for emotion detection and found that they are less susceptible to subjective interpretation than facial expressions or tones of voice.

### 4. Dimensional model for emotion recognition.
The dimensional model, which represents emotions in a coordinated space, has been widely used due to its intuitive nature. Several studies (Russell, 1980; Posner et al., 2005) have used this model and have highlighted its effectiveness in capturing the continuity and interrelationship of emotional experiences.

### 5. Deep learning in EEG emotion recognition
Deep learning algorithms have shown considerable success in various applications, including EEG emotion detection. Traditional machine learning methods often require manual feature extraction, which can be time consuming and error prone. However, deep learning provides automatic feature extraction and end-to-end classification, leading to better results (Cecotti & Graser, 2011; Bashivan et al., 2015).

The combination of BCI, emotion detection and regulation, EEG signals, dimensional modeling, and deep learning appears to be a promising avenue for further research. The next challenge is to effectively integrate these components to create a reliable and effective system for mental health applications.

## References
- Bashivan, P., Rish, I., Yeasin, M., & Codella, N. (2015). Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks. arXiv preprint arXiv:1511.06448.
- Cecotti, H., & Graser, A. (2011). Convolutional neural networks for P300 detection with application to brain-computer interfaces. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(3), 433-445.
- Gross, J. J. (2014). Emotion regulation: Conceptual and empirical foundations. In Handbook of emotion regulation (2nd ed., pp. 3-20). Guilford Press.
- Koelstra, S., Mühl, C., Soleymani, M., Lee, J. S., Yazdani, A., Ebrahimi, T., ... & Patras, I. (2012). DEAP: A database for emotion analysis; using physiological signals. IEEE Transactions on Affective Computing, 3(1), 18-31.
- Kret, M. E., & De Gelder, B. (2012). A review on sex differences in processing emotional signals. Neuropsychologia, 50(7), 1211-1221.
- Lin, Y. P., Wang, C. H., Jung, T. P., Wu, T. L., Jeng, S. K., Duann, J. R., & Chen, J. H. (2010). EEG-based emotion recognition in music listening. IEEE Transactions on Biomedical Engineering, 57(7), 1798-1806.
- Lotte, F., Bougrain, L., Cichocki, A., Clerc, M., Congedo, M., Rakotomamonjy, A., & Yger, F. (2018). A review of classification algorithms for EEG-based brain–computer interfaces: a 10-year update. Journal of neural engineering, 15(3), 031005.
- Posner, J., Russell, J. A., & Peterson, B. S. (2005). The circumplex model of affect: An integrative approach to affective neuroscience, cognitive development, and psychopathology. Development and psychopathology, 17(3), 715-734.
- Russell, J. A. (1980). A circumplex model of affect. Journal of personality and social psychology, 39(6), 1161.
- Wolpaw, J. R., Birbaumer, N., McFarland, D. J., Pfurtscheller, G., & Vaughan, T. M. (2002). Brain–computer interfaces for communication and control. Clinical neurophysiology, 113(6), 767-791.
