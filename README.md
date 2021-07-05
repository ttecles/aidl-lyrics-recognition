# Lyrics Recognition using Deep Learning Techniques

Final project for the [UPC Postgraduate Course Artificial Intelligence with Deep Learning](https://www.talent.upc.edu/ing/estudis/formacio/curs/310400/postgraduate-course-artificial-intelligence-deep-learning/), edition Spring 2021

Team: [Anne-Kristin Fischer](https://www.linkedin.com/in/anne-kristin-fischer-2ba9a158/), [Joan Prat Rigol](https://www.linkedin.com/in/joan-prat-rigol-290b7341/), [Eduard Rosés Gibert](https://www.linkedin.com/in/eduard-ros%C3%A9s-gibert-b87b9984/), [Marina Rosés Gibert](https://www.linkedin.com/in/marina-ros%C3%A9s-gibert-14057a195/)

Advisor: [Gerard I. Gállego](https://www.linkedin.com/in/gerard-gallego/)

To install the project we recommend using a virtual environment (venv). Steps to follow:

```bash 
sudo apt update
sudo apt install python3 python3-venv python3-dev
python3 -m venv .venv --prompt aidl-lyrics-recognition
source .venv/bin/activate
pip install -r requirements.txt
```

## Table of Contents <a name="toc"></a>
1. [Introduction](#intro)
    1. [Motivation](#motivation)
    2. [Hypothesis and project goals](#goals)
    3. [Milestones](#milestones)
2. [Data set](#dataset)
3. [Working Environment](#working_env)
4. [General Architecture](#architecture)
    1. [Main hyperparameters](#main_hyperparameters)
    2. [Evaluation metrics](#metrics)
5. [First Tests](#initial)
    1. [Preprocessing the data set](#dataset_preprocess)
    2. [Parameters](#parameters)
6. [Results](#results)
7. [Results improvement](#improving_results)
8. [The Google Cloud instance](#gcinstance)
9. [Conclusions](#conclusions)
10. [Next steps](#next_steps)
11. [References](#references)

## 1. Introduction <a name="intro"></a>
To this day few research is done in music lyrics recognition which is still considered a complexe task. For its approach two subtasks can be determined:

1. The singing voice needs to be extracted from the song by means of source separation. What seems to be an easy task for the human brain, remains a brain teaser for digital signal processing because of the complexe mixture of signals.
2. The second subtask aims to transcribe the obtained audio text of the singing voice into written text. This can be thought of as a speech recognition task. A lot of progress has been made for standard speech recognition tasks. Though, experiments with music made evident that the recognition of text of a singing voice is more complexe than pure speech recognition due to its increasing acoustical features.

Practical applications for music lyrics recognition such as the creation of karaoke versions or music information retrieval tasks motivate to tackle the aforementioned challenges.

<p align="right"><a href="#toc">To top</a></p>

### 1.1 Motivation <a name="motivation"></a>
Our decision for a lyrics recognition task with deep learning techniques is the attempt to combine several of our personal and professional interests. All team members have a more or less professional background in the music industry additionally to a particular interest in source separation tasks and natural language processing.

<p align="right"><a href="#toc">To top</a></p>

### 1.2 Hypothesis and Project Goals <a name="goals"></a>
* Extract the voice of a song and transcribe the lyrics with Demucs + Wav2Vec
* Analysis of results
* Deploy a web app for lyrics extraction
* Suggestions for further studies and investigation

<p align="right"><a href="#toc">To top</a></p>

### 1.3 Milestones <a name="milestones"></a>
To reach our goal, we set up the following milestones:
* Find a suitable data set
* Preprocess the data for its implementation into the model
* Define the model 
* Implement the model
* Train the model
* Analyse the obtained results 
* Implement the project inside a web application
* Make suggestions for further investigation

<p align="right"><a href="#toc">To top</a></p>

## 2. Data set <a name="dataset"></a>
To train our model we opted for the [DALI data set] (https://github.com/gabolsgabs/DALI), published in 2018. It is to this day the biggest data set in the field of singing voice research which aligns audio to notes and their lyrics along high quality standards. Access was granted to us for the first version, DALI v1, with 5358 songs in full duration and multiple languages. For more information please check as well [this article] (https://transactions.ismir.net/articles/10.5334/tismir.30/), published by the International Society for Music Information Retrieval.

<p ><img src="images/02-dali.jpg" width="300"></p>

<p align="right"><a href="#toc">To top</a></p>

## 3. Working Environment <a name="working_env"></a>
To develop the base model with 395 MM parameters, we used [Google Colab](https://colab.research.google.com/) as it was fast and easy for us to access. To train our model we made the first free tests with [wandb] (https://wandb.ai/site). For the full training with 580 MM parameters we then switched to a VM instance on [Google Cloud](https://cloud.google.com/).

<p ><img src="images/02-collab.jpg" width="200"> <img src="images/02-docker-logo.png" width="200"> <img src="images/03-wandb-logo.png" width="200"> <img src="images/02-googlecloud.jpg" width="200"></p>

<p align="right"><a href="#toc">To top</a></p>

## 4. General Architecture <a name="architecture"></a>
 Few research is done so far for music lyrics recognition in general and mostly spectrograms in combination with CNNs are used. In the context of this project we explore the possibility of a highly performing alternative by combining two strong models: the Demucs model for the source separation task in combination with a Wav2Vec model for the transcription task. Demucs is currently the best performing model for source separation based on waveform and so far the only waveform-based model which can compete with more commonly used spectrogram-based models. Wav2Vec is considered the current state-of-the-art model for automatic speech recognition.

 <p align="right"><a href="#toc">To top</a></p>

### 4.1 Main Hyperparameters <a name="main_hyperparameters"></a>
### 4.2 Evaluation Metrics <a name="metrics"></a>
For training and validation we opted for CTT loss.

## 5. First Tests <a name="initial"></a>
### 5.1 Preprocessing the data set <a name="dataset_preprocess"></a>
Preprocessing the data set correctly for our purpose was proven to be one of the major obstacles we encountered. We focused on songs in English only, that is 3491 songs in full duration. Preprocessing included omitting special characters as well as negative time stamps and transforming the lyrics in upper case only. To make sure to obtain meaningful results after training and to avoid cut-off lyrics, we prepared chunks. For these chunks we discarded words split among multiple notes at the beginning and end of each chunk and we cut out silent passages without voice. To make data accessible for our model, the audio waveform needed to be resampled to a sample rate of 44100 Hz.
As alignment is done automatically in DALI and groundtruth is available only for few audio samples, we followed the suggestions for train/validation/test split by the authors. That is

            | Correlations      | tracks (v1)
----------- | ----------------- | ---------------------
Test        |NCCt >= .94        | 1.0: 167
Validation  |.94 > NCCt >= .925 | 1.0: 423
Train       |.925 > NCCt >= .8  | 1.0: 4768

where NCCt is a correlation score which indicates how accurate the autmatic alignment is. Higher means better. The number of tracks refers to the whole data set, including as well songs in other languages.

### 5.2 Finding the right parameters <a name="parameters"></a>
## 6. Results <a name="results"></a>
## 7. Results improvement <a name="improving_results"></a>
## 8. Google Cloud instance <a name="gcinstance"></a>

## 9. Conclusions <a name="conclusions"></a>
Today as then the community laments a lack of well structered, aligned, large data sets for music information retrieval tasks.

<p align="right"><a href="#toc">To top</a></p>

## 10. Next steps <a name="next_steps"></a>
Further research could be done for:
* melody extraction
* chords transcription
* adding a language model to improve the results of the transcription task
* summary of the lyrics
* pitch recognition
* contribute to larger datasets of high quality

<p align="right"><a href="#toc">To top</a></p>

## 11. References <a name="references"></a>
https://towardsdatascience.com/wav2vec-2-0-a-framework-for-self-supervised-learning-of-speech-representations-7d3728688cae
https://ieeexplore.ieee.org/abstract/document/5179014
https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1318.pdf
https://ieeexplore.ieee.org/document/6682644?arnumber=6682644
https://www.researchgate.net/publication/42386897_Automatic_Recognition_of_Lyrics_in_Singing
https://europepmc.org/article/med/20095443
https://asmp-eurasipjournals.springeropen.com/articles/10.1155/2010/546047
https://arxiv.org/abs/2102.08575