# Classification of 12-lead ECGs from the PTB-XL Database by Yap Jun Hong (Nemo)

## Table of contents

1. [Section 1: Project directory](#section-1-project-directory)
2. [Section 2: Why is this project important?](#section-2-why-is-this-project-important)
3. [Section 3: Dataset used](#section-3-dataset-used)
4. [Section 4: Goals and metrics of this project](#section-4-goals-and-metrics-of-this-project)
5. [Section 5: Data exploration and engineered features](#section-5-data-exploration-and-engineered-features)
6. [Section 6: References](#section-6-references)

## Section 1: Project directory

### Part 1.1: Notebooks in this project

1. [Raw Signals to Dataframe](code/01_raw_signals_to_dataframe.ipynb)
2. [Data Analysis](code/02_data_analysis.ipynb)
3. [Feature Engineering](code/03_feature_engineering.ipynb)
4. [Cascading Classifier](code/04a_cascading_classifiers_modelling.ipynb)
5. [MTEX-CNN](code/04b_mtex_cnn_classification.ipynb)

To run this project, start from notebook 2, Data Analysis, onwards. You cannot run notebook 1 unless you download the PTB-XL data from Kaggle or Physionet and put it in a folder named 'ptb_xl' under the data folder. You only need the 100Hz version of the data, meaning you only need to keep the 'records100' folder in the dataset you download.

I have saved a processed version of the raw signals, hosted by Github-lfs (Large File Storage).

If you need help understanding ECG data, refer to the markdown file [ECG Basics and Interpretation](./ecg_basics/ecg_basics_and_interpretation.md), where I explain the basics of interpreting an ECG, the libraries used in Python, and the basics of time-series data.

### Part 1.2: Libraries used in this project

This project makes extensive use of [Neurokit2](https://pypi.org/project/neurokit2/), which is commonly used for processing bio-signals, including ECGs.

To access the files, this project also uses [Abstract Syntax Trees (ast)](https://docs.python.org/3/library/ast.html) and [WaveForm DataBase (wfdb)](https://pypi.org/project/wfdb/)

## Section 2: Why is this project important?

### Part 2.1: What is an ECG?

ECG is short for an "**E**lectro**c**ardio**g**ram", a tool frequently used by doctors to measure the strength and flow of electrical currents in the heart. Abnormalities in these electrical signals correspond to disease symptoms that doctors can recognise and address. The most common type of ECG used is a '12-lead ECG', which can be thought of as viewing the electrical signal flow in the heart from 12 different angles or viewpoints. The 12-lead ECG is measured by sticking electrodes on the limbs, for a vertical view, and on the chest, for a horizontal view. The 12-lead ECG gives information such as:

- Strength of the generated electrical signal
- Direction of the flow of the electrical signal through the heart

A reading from a healthy patient is called 'Sinus Rhythm'.

### Part 2.2: How can machine learning help doctors with reading ECG data?

Manually interpreting an ECG requires years of training on the job and in medical school. Manual interpretation is also time-consuming; doctors may make mistakes over the course of a day as they tire out. Using machine learning methods to automatically interpret ECG data can help increase the chances of noticing mistakes. In emergency situations, automatic interpretation can also be used in areas and emergencies without sufficiently skilled doctors. However, the overall goal of automatic interpretation is to assist doctors in coming to conclusions about abnormalities in the ECG.

## Section 3: Dataset used

### Part 3.1: Dataset sources

This project uses the PTB-XL database to do classification. The dataset can be found both on [physionet](https://www.physionet.org/content/ptb-xl/1.0.1/) and [kaggle](https://www.kaggle.com/bjoernjostein/ptbxl-electrocardiography-database). A summary of the dataset is provided below.

|**Database**|**Number of patients**|**Number of recordings**|**Number of leads (channels)**|**Duration of each recording (seconds)**|
|---|---|---|---|---|
|PTB-XL|18885|21837|12|10|

### Part 3.2: Dataset properties

This part talks about general properties of the dataset.

1. The ECG recordings were obtained in a hospital or clinical setting over nearly 7 years between October 1989 and June 1996. Devices from Schiller AG were used.

2. Clinicians used a sampling frequency of 500Hz. The data was saved in WaveForm DataBase (WFDB) format with 16-bit precision at a resolution of 1 μV/LSB. The dataset includes a downsampled version of the waveform data at a sampling frequency of 100Hz.

Relevant metadata is stored in 'ptbxl_database.csv', with one row per record identified by ecg_id. The csv file contains 28 columns. These columns can be categorised into:

3. **Identifiers**: Each record is identified by a unique ecg_id. The corresponding patient is encoded via patient_id. The paths to the original record (500 Hz) and a downsampled version of the record (100 Hz) are stored in filename_hr and filename_lr.

4. **General Metadata**: demographic and recording metadata such as age, sex, height, weight, nurse, site, device and recording_date

5. **ECG statements**: core components are scp_codes (SCP-ECG statements as a dictionary with entries of the form statement: likelihood, where likelihood is set to 0 if unknown) and report (report string). Additional fields are heart_axis, infarction_stadium1, infarction_stadium2, validated_by, second_opinion, initial_autogenerated_report and validated_by_human.

6. **Signal Metadata**: signal quality such as noise (static_noise and burst_noise), baseline drifts (baseline_drift) and other artifacts such as electrodes_problems. The database engineers also provide extra_beats for counting extra systoles and pacemaker for signal patterns indicating an active pacemaker.

7. **Cross-validation Folds**: recommended 10-fold train-test splits (strat_fold) obtained via stratified sampling while respecting patient assignments, i.e. all records of a particular patient were assigned to the same fold. Records in fold 9 and 10 underwent at least one human evaluation and are therefore of a particularly high label quality. The data engineers proposed to use folds 1-8 as training set, fold 9 as validation set and fold 10 as test set.

Points 3-7 were taken directly from the physionet page's description with slight modification.

## Section 4: Goals and metrics of this project

### Part 4.1: Primary and secondary goals

The primary goal of this project is to classify the ECG data into 6 classes, 1 NORM class and 5 other classes representing diseased states. This will be achieved in two ways: 

- First, through a Cascading Classifier comprised of a Random Forest and Deep Neural Network with 5 hidden layers.

- Second, through a neural network architecture called a MTEX-CNN. The advantage of this MTEX-CNN is that it is made for multi-channel time series, which is what our data is made of. It can also be explainable with a grad-CAM (grad Class Activation Map) layer, although this project could not implement the explainability aspect of it. 

All information from the 12 leads will be used, as inspired by the Physionet challenge 2020 from which I learnt of this dataset. Future projects can improve on this by using dimensionality reduction of PCA techniques and comparing them with 2-lead ECG data.

The secondary goals with this project are:

- To learn to read scientific papers and learn to implement architectures as described in the papers. 

- To learn to process and work with Bio-signals, in this case, ECG data.

### Part 4.1: Metrics used

Accuracy was the main metric used. For MTEX-CNN, AUC and F1-score were used too for MTEX-CNN to compare with benchmarks.

### Part 4.2: Were the goals achieved?

The primary goal was achieved with an accuracy of 69% by the MTEX-CNN architecture and 60% with the Cascading Classifier. The training and testing data split used were the ones set by the dataset designers. MTEX-CNN had an AUC score of 84%, with an fmin score of 0.09 for class 1 ('HYP' class, a class representing a diseased heart) and fmax score of 0.80 for class 3 ('NORM' class). 

The secondary goals were partially achieved. While I learnt about the different domains that time-series signals could take, alongside other ideas like multiple hypothesis testing (for testing stationarity of the signals), I could only partially achieve implementing architectures in the papers. This was partially because of being unable to find specific descriptions of actions done, such as linearly combining features. Another reason is because simply using a Cascading Classifier was new to me, and implementing what the authors did with the Cascading Classifier specifically was slightly beyond my understanding.

I also had to learn how to make my code efficient. Early attempts to apply the Neurokit2 library functions to the dataframe took incredibly long to run. These were done with for loops. I had to use map/apply after reshaping the dataframe for it to run within 5-10 seconds.

### Part 4.3: What I could improve in the future

I did not include meta-data inside the project. When I first started this project, I had only thought of using ECG data as input and did not include meta-data like statements by the doctors. My future projects with bio-signal data will try to include meta-data inside.

I linearly combined the features by simply adding them up to give one number for the ECG data. I learnt, towards the end of the project, that it was possible to hold an array of data in a single cell in a dataframe. Future projects could try to include an array of numbers for each bio-signal data.

I do not think I implemented the Cascading Classifier properly. Future iterations of this project will look into further refining this classifier.

## Section 5: Data exploration and engineered features

### Part 5.1: Data exploration

Data exploration was attempted, although a standardised process of exploring ECG data is hard to find. Most papers where ECG data is classified did not have a section where the ECG data was explored.

In my exploration, I decomposed the ECG signal into its trend, seasonality, and residue. I checked for stationarity, and most of the ECGs (89%) were considered stationary. I checked for P, Q, R, S, and T peaks, and checked heart rate variability, although I was not sure how to interpret heart rate variability.

### Part 5.2: Engineered features

The features engineered roughly follow what was created in the Cascading Classifier paper. They can be split into 3 types of features:

Statistical features: Kurtosis, used to detect signal quality, and Skew, used to detect symmetry around the R-peak.

Power and Frequency features: Fast-fourier transformed data (frequency domain), for phase and power features, and Short-time fourier transformed data (time-frequency domain), for frequency and location features.

Amplitude/Voltage features: Max, Min, and Avg peak heights, as the amplitude on the ECG paper represents the amount of voltage.

These features were fed into a Logistic Regression in an attempt to linearly combine them, before being fed into the Cascading Classifier. However, this produced very slightly worse results, less than 1%, compared to being fed directly into the Cascading Classifier.

For the MTEX-CNN architecture, the raw signals were fed inside with no changes made to them.

## Section 6: References

### Part 6.1: The papers referred to for classification

Two papers were referred to for classification

[1] G. Chen, Z. Hong, Y. Guo and C. Pang, "A cascaded classifier for multi-lead ECG based on feature fusion,"_Computer Methods and Programs in Biomedicine_, vol. 178, pp. 135-143, September 2019. Available: [https://doi.org/10.1016/j.cmpb.2019.06.021](https://doi.org/10.1016/j.cmpb.2019.06.021) [Accessed Feb. 24, 2021].

[2] R. Assaf, I. Giurgiu, F. Bagehorn and A. Schumann, "MTEX-CNN: Multivariate Time Series EXplanations for Predictions with Convolutional Neural Networks," _2019 IEEE International Conference on Data Mining (ICDM), Beijing, China, 2019_, pp. 952-957. Available: [10.1109/ICDM.2019.00106.](https://ieeexplore.ieee.org/document/8970899) [Accessed Feb. 24, 2021].

A third was referred to for the exact architecture of MTEX-CNN, which also critiques MTEX-CNN and offers its own explainable architecture

[3] K. Fauvel, T. Lin, V. Masson, E. Fromont and A. Termier, "XCM: An Explainable Convolutional Neural Network for Multivariate Time Series Classification," 2020, _arXiv:2009.04796v2_. Available: [arXiv:2009.04796v2](https://arxiv.org/abs/2009.04796) [Accessed Feb. 24, 2021].

### Part 6.2: Library used

This project uses Neurokit2 to process bio-signals.

[4] Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lesspinasse, F., Pham, H., Schölzel, C., & S H Chen, A. (2020). "NeuroKit2: A Python Toolbox for Neurophysiological Signal Processing." Retrieved February 24, 2021, from https://github.com/neuropsychology/NeuroKit

### Part 6.3: PTB-XL dataset

This project uses the PTB-XL dataset...

[5] Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.1). PhysioNet. https://doi.org/10.13026/x4td-x982.

[6] Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T. (2020), PTB-XL: A Large Publicly Available ECG Dataset. Scientific Data. https://doi.org/10.1038/s41597-020-0495-6.

...Which you can retrieve from Physionet.

[7] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
