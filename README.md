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
- [Surveying and Analyzing existing literature](#surveying-and-analyzing-existing-literature)
- [References](#references)

## Introduction
Emotion recognition stands as a frontier topic in the interdisciplinary domain of psychology, neuroscience, and artificial intelligence, serving as a gateway to the next generation of human-computer interaction, and unlocking potential applications that range from healthcare, gaming, marketing to more personalized and responsive artificial intelligence systems. In this context, one promising modality for emotion recognition that has attracted significant attention is the use of Electroencephalogram (EEG).

As a neuroimaging technique, EEG captures the electrical activity of the brain, offering a direct interface with the neuronal signals that correspond to various cognitive states, including emotions. However, EEG signals are complex and often require advanced processing and analytical techniques to decipher the subtle patterns that correlate with different emotional states.

Amid this backdrop, this research leverages the SEED dataset, which comprises EEG recordings under three emotional states - Negative, Positive, and Neutral. This dataset, known for its robustness and validity in the context of EEG-based emotion recognition, provides a valuable foundation for our exploration.
To unravel the intricacies of these EEG signals, this study employs an array of signal processing techniques such as Independent Component Analysis (ICA) for artifact removal, Wavelet Energy, and Shannon Entropy for feature extraction, and Principal Component Analysis (PCA) for dimensionality reduction.

The extracted features are then used to train and test four different machine learning and deep learning models - Recurrent Neural Networks (RNN), Convolutional Recurrent Neural Networks (CRNN), two-layer Bidirectional Long Short-Term Memory (BiLSTM), and Convolutional Neural Network - BiLSTM (CNN-BiLSTM). These models were chosen due to their proven capability in handling sequence data and their robustness in various classification tasks.

Through a rigorous and systematic study, this research aims to explore the effectiveness of these models in classifying emotions based on EEG data and contribute to the broader goal of advancing EEG-based emotion recognition.


Objectives:

-	To explore the feasibility of EEG-based emotion recognition using the SEED dataset.
-	To employ Independent Component Analysis (ICA) for artifact removal in the EEG dataset.
-	To apply Wavelet Energy and Shannon Entropy on the artifact-free EEG data for feature extraction.
-	To implement Principal Component Analysis (PCA) for dimensionality reduction.
-	To train and test four distinct ML and DL models, namely Recurrent Neural Networks (RNN), Convolutional Recurrent Neural Networks (CRNN), two-layer Bidirectional Long Short-Term Memory (BiLSTM), and Convolutional Neural Network - BiLSTM (CNN-BiLSTM) for emotion classification.
-	To compare and analyze the performance of these models to determine the most efficient model for EEG-based emotion recognition.


## Demo Video


## Dataset

Dataset (SEED) 
SEED dataset provided by the Brain-like Computing & Machine Intelligence (BCMI) laboratory, which is led by Prof. Bao-Liang Lu. To gain access to the SEED IV dataset, an application is required. Please visit the BCMI laboratory’s website to apply for access before using this dataset.

Download dataset in this link https://bcmi.sjtu.edu.cn/home/seed/
BCMI Laboratory homepage https://bcmi.sjtu.edu.cn/

Fifteen Chinese film clips were carefully chosen through a preliminary study, which tended to include emotions such as <b>positive</b>, <b>neutral</b>, and  <b>negative</b>. A total of 15 subjects participated in the experiment.

The experiment employed a 62-Channel ESI NeuroScan system and was designed as follows: each participant underwent three sessions on different days, with each session containing 24 trials. In each trial, the participant watched a film clip intended to induce a specific emotional state (positive, neutral, negative). While the participant was watching the clip, their EEG signals and eye movements were recorded.
<br>
<p align="center">
<img src="./imgs/raw dataset lcoations.jpg" alt="raw dataset" >
  </p>
<br>
 
The dataset comprises two files: eeg_raw_data and eye_raw_data. For the purpose of this experiment, we focused only on the eeg_raw_data. The eeg_raw_data folder contains the raw EEG signals from the 15 participants. Within eeg_raw_data, there are three folders named 1, 2, and 3, corresponding to the three sessions. Each .mat file is named in the format {subjectName}_{Date}.mat. These folders store a structure with fields named "cz_eeg1", "cz_eeg2", up to "cz_eeg24", which correspond to the EEG signal recorded during the 24 trials. For each of the signal processing, the raw EEG data are first downsampled to a 200 Hz sampling rate.
<br>
<p align="center">
<img src="./imgs/experiment framework.jpg" alt="experiment framework v2" >
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

###  Analyzing the hidden components within EEG using ICA with ICLabel
The analysis commenced with the loading of the EEG dataset into EEGLAB, followed by the visualization of the raw Independent Components (ICs) through the ICLabel plugin. This step served as an initial examination of the data before any pre-processing. Subsequently, a basic filter was applied to eliminate noise and confine the data to relevant frequencies. This filtered data was then plotted for visual inspection and comparison against the raw data. The next phase in the data processing involved Artifact Subspace Reconstruction (ARS), which is aimed at removing artifacts by reconstructing the EEG data. Post ARS, the ICLabel plugin was employed again to visualize the improved data quality. For a more granular analysis, attention was focused on channels FP1, FP2, and FP3. By juxtaposing the raw, filtered, and ARS-corrected plots of these channels, a subtle yet significant enhancement in data quality was discernible. The application of filtering and ARS correction reduced noise and rendered the EEG signals cleaner, which is instrumental for the accurate analysis of brain activity. This streamlined process underscores the importance of methodical preprocessing in EEG analysis, ensuring that the data is pruned and primed for further investigation.

<p>
  <table style="padding: 10px; border: solid 1px black">
  <tr>
    <td>&nbsp;</td>
    <td colspan="2" style="text-align: center; padding: 5px; font-weight: 600">
      Pre-processing
    </td>
    <td colspan="7" style="text-align: center; padding: 5px; font-weight: 600">
      Average numbers of ICs classified by ICLabel (FP1, FP2, F3)
    </td>
  </tr>
  <tr>
    <td>EEG (14 Channels & Eyeblink Dataset)</td>
    <td style="text-align: center; padding: 5px">bandpass filter</td>
    <td style="text-align: center; padding: 5px">ASR</td>
    <td style="text-align: center; padding: 5px">Brain</td>
    <td style="text-align: center; padding: 5px">Muscle</td>
    <td style="text-align: center; padding: 5px">Eye</td>
    <td style="text-align: center; padding: 5px">Heart</td>
    <td style="text-align: center; padding: 5px">Line Noise</td>
    <td style="text-align: center; padding: 5px">Channel Noise</td>
    <td style="text-align: center; padding: 5px">Other</td>
  </tr>
  <tr>
    <td>Raw</td>
    <td style="text-align: center; padding: 5px">&nbsp;</td>
    <td style="text-align: center; padding: 5px">&nbsp;</td>
    <td style="text-align: center; padding: 5px">FP1=76.6% <br> FP2=51.3% <br> F3=42.1%</td>
    <td style="text-align: center; padding: 5px">FP1=0.9% <br> FP2=2.2% <br> F3=2.2%</td>
    <td style="text-align: center; padding: 5px">FP1=0.1% <br> FP2=0.2% <br> F3=0.2%</td>
    <td style="text-align: center; padding: 5px">FP1=0.6% <br> FP2=0.1% <br> F3=0.3%</td>
    <td style="text-align: center; padding: 5px">FP1=3.7% <br> FP2=1.8% <br> F3=5.2%</td>
    <td style="text-align: center; padding: 5px">FP1=0.8 <br> FP2=0.1% <br> F3=0.3%</td>
    <td style="text-align: center; padding: 5px">FP1=17% <br> FP2=44.3% <br> F3=49.7%</td>
  </tr>
  <tr>
    <td>Filtered</td>
    <td style="text-align: center; padding: 5px">v</td>
    <td style="text-align: center; padding: 5px">&nbsp;</td>
    <td style="text-align: center; padding: 5px">4</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">10</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
  </tr>
  <tr>
    <td>ASR-corrected</td>
    <td style="text-align: center; padding: 5px">v</td>
    <td style="text-align: center; padding: 5px">v</td>
    <td style="text-align: center; padding: 5px">6</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">6</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">0</td>
    <td style="text-align: center; padding: 5px">2</td>
  </tr>
</table>
  
</p>

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
