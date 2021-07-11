# Lyrics Recognition using Deep Learning Techniques

Final project for the [UPC Postgraduate Course Artificial Intelligence with Deep Learning](https://www.talent.upc.edu/ing/estudis/formacio/curs/310400/postgraduate-course-artificial-intelligence-deep-learning/), edition Spring 2021

Team: [Anne-Kristin Fischer](https://www.linkedin.com/in/anne-kristin-fischer-2ba9a158/), [Joan Prat Rigol](https://www.linkedin.com/in/joan-prat-rigol-290b7341/), [Eduard Rosés Gibert](https://www.linkedin.com/in/eduard-ros%C3%A9s-gibert-b87b9984/), [Marina Rosés Gibert](https://www.linkedin.com/in/marina-ros%C3%A9s-gibert-14057a195/)

Advisor: [Gerard I. Gállego](https://www.linkedin.com/in/gerard-gallego/)

GitHub repository: [https://github.com/ttecles/aidl-lyrics-recognition](https://github.com/ttecles/aidl-lyrics-recognition)

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
    2. [Project Goals](#goals)
    3. [Milestones](#milestones)
2. [Data Set](#dataset)
3. [Working Environment](#working_env)
4. [General Architecture](#architecture)
    1. [Main Hyperparameters](#main_hyperparameters)
    2. [Evaluation Metrics](#metrics)
5. [Preprocessing and tests](#tests)
    1. [Preprocessing the data set](#dataset_preprocess)
    2. [Fine-tuning of the parameters](#parameters)
6. [Results and results improvement](#results)
    1. [Experiment 1: First train with the full dataset](#experiment_1)
    2. [Experiment 2: Overfitting with one chunk](#experiment_2)
    3. [Experiment 3: Awesome experiment](#experiment_3)
7. [Conclusions](#conclusions)
8. [Next Steps](#next_steps)
9. [References](#references)

## 1. Introduction <a name="intro"></a>
To this day few research is done in music lyrics recognition which is still considered a complex task. For its approach two subtasks can be determined:

1. The singing voice needs to be extracted from the song by means of source separation. What seems to be an easy task for the human brain, remains a brain teaser for digital signal processing because of the complexe mixture of signals.
2. The second subtask aims to transcribe the obtained audio text of the singing voice into written text. This can be thought of as a speech recognition task. A lot of progress has been made for standard speech recognition tasks. Though, experiments with music made evident that the recognition of text of a singing voice is more complex than pure speech recognition due to its increasing acoustical features.

Practical applications for music lyrics recognition such as the creation of karaoke versions or music information retrieval tasks motivate to tackle the aforementioned challenges.
<p align="right"><a href="#toc">To top</a></p>

### 1.1 Motivation <a name="motivation"></a>
Our decision for a lyrics recognition task with deep learning techniques is the attempt to combine several of our personal and professional interests. All team members have a more or less professional background in the music industry additionally to a particular interest in source separation tasks and natural language processing.

<p align="middle"><a href="https://drive.google.com/uc?export=view&id=1k1CzCI42BNfLrkkPh_0qoyXq_-fJic3j"><img src="https://drive.google.com/uc?export=view&id=1k1CzCI42BNfLrkkPh_0qoyXq_-fJic3j" style="width: auto; max-width: 50%; height: 100px" title="motivation" /></p>           
       
_Figure 1<span>: Our passion for music, language and deep learning combined\.</span>_

<p align="right"><a href="#toc">To top</a></p>

### 1.2 Project Goals <a name="goals"></a>
* Extraction of the voice of a song and transcription of the lyrics with Demucs and Wav2Vec models
* Analysis of the results
* Deployment of a web application for lyrics extraction
* Suggestions for further studies and investigation
<p align="right"><a href="#toc">To top</a></p>

### 1.3 Milestones <a name="milestones"></a>
To reach our goals, we set up the following milestones:
* Find a suitable data set
* Preprocess the data for its implementation into the model
* Define the model 
* Implement the model
* Train the model
* Analyse the obtained results 
* Implement the project inside a web application
* Make suggestions for further investigation
* _Optional: add a language model to improve the results of the transcription task_
<p align="right"><a href="#toc">To top</a></p>

## 2. Data Set <a name="dataset"></a>
To train our model we opted for the [DALI data set](https://github.com/gabolsgabs/DALI), published in 2018. It is to this day the biggest data set in the field of singing voice research which aligns audio to notes and their lyrics along high quality standards. Access was granted to us for the first version, DALI v1, with 5358 songs in full duration and multiple languages. For more information please check as well [this article](https://transactions.ismir.net/articles/10.5334/tismir.30/), published by the International Society for Music Information Retrieval.

<p align="middle"><a href="https://drive.google.com/uc?export=view&id=1cs0GjeBhxCCY2mSCqbnSrK9lI0XxjhF1"><img src="https://drive.google.com/uc?export=view&id=1cs0GjeBhxCCY2mSCqbnSrK9lI0XxjhF1" style="width: auto; max-width: 100%; height: 200px" title="dali_alignment" /></p>
    
_Figure 2: Alignment of notes and text in DALI data set based on triples of {time (start and duration), note, text}_
    
<p align="middle"><a href="https://drive.google.com/uc?export=view&id=1wiIgKXR0aWBqDtang5Xaim9tqUS06sTv"><img src="https://drive.google.com/uc?export=view&id=1wiIgKXR0aWBqDtang5Xaim9tqUS06sTv" style="width: auto; max-width: 100%; height: 200px" title="dali_horizontal" /></p>
    
_Figure 3: Horizontal granularity in DALI data set where paragraphs, lines, words and notes are interconnected vertically_

<p align="right"><a href="#toc">To top</a></p>

## 3. Working Environment <a name="working_env"></a>
To develop the base model with 395 MM parameters, we used [Google Colab](https://colab.research.google.com/) as it was fast and easy for us to access. For visualization of the results and to train our model we made the first free tests with [wandb](https://wandb.ai/site). For the full training with 580 MM parameters we then switched to a VM instance with one V100 GPU on [Google Cloud](https://cloud.google.com/). [PyTorch](https://pytorch.org/) is used as the overall framework.

<p align="middle"><a href="https://drive.google.com/uc?export=view&id=1YnUwkz5QRjbJ3d3inmizqeO3kYA_WcBL"><img src="https://drive.google.com/uc?export=view&id=1YnUwkz5QRjbJ3d3inmizqeO3kYA_WcBL" style="width: auto; max-width: 50%; height: 80px" title="Colab" /> <a href="https://drive.google.com/uc?export=view&id=1_hBcgu2pRQETfRexso92teeKkfmZX-sQ"><img src="https://drive.google.com/uc?export=view&id=1_hBcgu2pRQETfRexso92teeKkfmZX-sQ" style="width: auto; max-width: 50%; height: 80px" title="wandb" /> <a href="https://drive.google.com/uc?export=view&id=1s4UkYQ5tWJ22L24AiFqjn9KGFd-l-6cF"><img src="https://drive.google.com/uc?export=view&id=1s4UkYQ5tWJ22L24AiFqjn9KGFd-l-6cF" style="width: auto; max-width: 50%; height: 80px" title="GCloud" /> <a href="https://drive.google.com/uc?export=view&id=1IouSQvK4_ibRmmvdc_nd-fbl1bHvIw7Z"><img src="https://drive.google.com/uc?export=view&id=1IouSQvK4_ibRmmvdc_nd-fbl1bHvIw7Z" style="width: auto; max-width: 50%; height: 80px" title="pytorch" /></p >
<p align="right"><a href="#toc">To top</a></p>

## 4. General Architecture <a name="architecture"></a>
Few research is done so far for music lyrics recognition in general and mostly spectrograms in combination with CNNs are used. In the context of this project we explore the possibility of a highly performing alternative by combining two strong models: the Demucs model for the source separation task in combination with a Wav2Vec model for the transcription task. Demucs is currently the best performing model for source separation based on waveform and so far the only waveform-based model which can compete with more commonly used spectrogram-based models. Wav2Vec is considered the current state-of-the-art model for automatic speech recognition. Additionally, we implemented KenLM as a language model on top to improve the output of the transcription task.

![image](https://drive.google.com/uc?export=view&id=1LGxdJUxxW76P5Kx58ALWSDbY23WUntNJ)
    
<p align="right"><a href="https://drive.google.com/uc?export=view&id=1fkhAahhfkPNFG-J4BxjxOLnXTDodUv8r"><img src="https://drive.google.com/uc?export=view&id=1fkhAahhfkPNFG-J4BxjxOLnXTDodUv8r" style="width: auto; max-width: 100%; height: 250px" title="Demucs" /> <a href="https://drive.google.com/uc?export=view&id=1HvDAh3QXVgdHqbUupv_Du3V5raWcZ2dK"><img src="https://drive.google.com/uc?export=view&id=1HvDAh3QXVgdHqbUupv_Du3V5raWcZ2dK" style="width: auto; max-width: 100%; height: 250px" title="wav2vec" /><p > 

![image](https://drive.google.com/uc?export=view&id=1GRGI8rrg4noZMoFKxSjcshsHOj1jD_3z)    
    _Figure 4: Overall model architecture with detailed insides in Demucs and Wav2Vec architecture and the KenLM language model on top_
<p align="right"><a href="#toc">To top</a></p>

### 4.1 Main Hyperparameters <a name="main_hyperparameters"></a>
    
For first training we applied standard values for our model hyperparameters, that is parameters which are proven to deliver first good results such as Adam optimizer and a learning rate of 0.0001. 
    
| Parameter | Demucs  | Wav2Vec  | KenLM (n-gram)
| ------------ | ------------ | ------------- | ----
| Optimizer | Adam | Adam |  
| Learning rate | 0.0001 | 0.0001 |
| Weight decay | 0.0001 | 0.0001 |
| Audio length | 10 sec. | 10 sec. |
| Batch Size |   |   |  
| Epochs |   |   |  
   
<p align="right"><a href="#toc">To top</a></p>

### 4.2 Evaluation Metrics <a name="metrics"></a>
For training and validation of Demucs and Wav2Vec model we opted for the CTC loss function (Connectionist Temporal Classification). CTC loss is most commonly used for speech recognition tasks, but can be applied as well to our sequence problem of audio recognition. The input sequence can be a spectrogram or, like in our case, in waveform. The sequence input is then fed into a RNN model, like our Demucs LSTM model. For KenLM we applied WER loss. The word error rate is obtained as a percentage dividing the total of wrongly predicted word (that is insertions, deletions and substitutions) by the total of the documents words.

<p align="middle"><a href="https://drive.google.com/uc?export=view&id=1XmH6hv-9iC0u5k-a01VmuAzQZM4vGz5M"><img src="https://drive.google.com/uc?export=view&id=1XmH6hv-9iC0u5k-a01VmuAzQZM4vGz5M" style="width: auto; max-width: 100%; height: 450px" title="ctc_loss_1" /> <a href="https://drive.google.com/uc?export=view&id=15KBnAoTLgT2WHMrjUrZWIFNFYu3m6to0"><img src="https://drive.google.com/uc?export=view&id=15KBnAoTLgT2WHMrjUrZWIFNFYu3m6to0" style="width: auto; max-width: 100%; height: 80px" title="ctc_loss_2" /></p>
    
_Figure 5: CTC loss: architecture and its calculation_
    
![image](https://drive.google.com/uc?export=view&id=1N4Ecf8kh_TDR_IatFsqR6TQBTHqyqQ86)      

_Figure 6: WER loss_    

<p align="right"><a href="#toc">To top</a></p>

## 5. Preprocessing and tests <a name="tests"></a>
### 5.1 Preprocessing the data set <a name="dataset_preprocess"></a>
Preprocessing the data set correctly for our purpose was proven to be one of the major obstacles we encountered. We focused on songs in English only, that is 3491 songs in full duration. Preprocessing included omitting special characters as well as negative time stamps and transforming the lyrics in upper case only. To make sure to obtain meaningful results after training and to avoid cut-off lyrics, we prepared chunks. For these chunks we discarded words split among multiple notes at the beginning and end of each chunk and we cut out silent passages without voice. To make data accessible for our model, the audio waveform needed to be resampled to a sample rate of 44100 Hz.
As alignment is done automatically in DALI and groundtruth is available only for few audio samples, we followed the suggestions for train/validation/test split by the authors. That is:

![image](https://drive.google.com/uc?export=view&id=17tIQ9EroDUCo4dG-1tF6OZDlSVmv5aii)    
    _Figure 7: Suggested NCCt scores for train, validation and test_
    
where NCCt is a correlation score which indicates how accurate the automatic alignment is. Higher means better. The number of tracks refers to the whole data set, including as well songs in other languages for both the first and second version of the dataset.

![image](https://drive.google.com/uc?export=view&id=1tDukLCKRWCIfKMtsCoh-WGEUI7jFvv5V)    
    _Figure 8: Automatic alignment of singing voice and text in DALI with teacher-student paradigm based on NCCt score for the student_
<p align="right"><a href="#toc">To top</a></p>

### 5.2 Fine-tuning of the parameters <a name="parameters"></a>
    
Parameter | Demucs  | Wav2Vec  | KenLM (n-gram)
--------- | ------- | -------- | --------------
Optimizer | Adam | Adam |  
Learning rate | 0.0001 | 0.0001 |
Weight decay | 0.0001 | 0.0001 |
Audio length | 10 sec. | 10 sec. |
Batch Size |   |   |  
Epochs |   |   |  

<p align="right"><a href="#toc">To top</a></p>

## 6. Results and results improvement <a name="results"></a>
To our surprise we obtained initially a negative loss which could be explained by the training of data slices containing no lyrics. Furthermore, one of our training runs showed the level of corruption for Demucs: the voice quality, epoch by epoch, got worse. Considering different learning rates and optimizers for Demucs and Wav2Vec proved to be reasonable as Wav2Vec needed more attention in terms of fine-tuning than already pretrained Demucs. To make sure our model was working, a sanity check came in handy where we tested the model on a small batch on its possibility to overfit. We gradually augmented the batch size using a controllable small dataset with a NCCt score higher than 0.95 to make sure our model would still train properly.
<p align="right"><a href="#toc">To top</a></p>
    
### 6.1 Experiment 1: First train with the full dataset <a name="experiment_1"></a> 
    
Step | Comments
--------- | ------
Hypothesis |  Our model will output awesome lyrics predictions.
Set up | 
Results | Our model shows weird metrics.
Conclusions |  We are not sure if our model is even training.
Links | [Run](https://wandb.ai/aidl-lyrics-recognition/demucs+wav2vec/runs/mhujqbrx?workspace=user-akifisch), [Report](https://wandb.ai/aidl-lyrics-recognition/demucs+wav2vec/reports/First-run-with-full-dataset--Vmlldzo4NDEzNjM)
    
<p align="right"><a href="#toc">To top</a></p>
    
### 6.2 Experiment 2: Overfitting with one chunk <a name="experiment_2"></a>
    
Step | Comments
--------- | ------
Hypothesis | Our model works if it is “able” to overfit.
Set up | 
Results | Our model overfits.
Conclusions | Our model is working and actually training.
Links | [Run](https://wandb.ai/aidl-lyrics-recognition/demucs+wav2vec/runs/1nofaz64?workspace=user-akifisch), [Report](https://wandb.ai/aidl-lyrics-recognition/demucs+wav2vec/reports/Overfit-1-chunk--Vmlldzo4NDEzNzc?accessToken=te8rgaea48t9a6rhaa2y15ymhfwurxnvnkcn0axjiqjew14e9d6i96re4ngqxdl5), [Audio track](https://drive.google.com/file/d/1-GKAjg45Fm3DNuVY8_A0AgwmUwDdGiNy/view?usp=sharing)

![image](https://drive.google.com/uc?export=view&id=1fI5c9Dob0yS7VtYOFcbNEVyn6VhST0Tm) ![image](https://drive.google.com/uc?export=view&id=1Wzr_bDE01l8Zb66tteI6_H36IeMS8Axb)
    
<p align="right"><a href="#toc">To top</a></p>
    
### 6.3 Experiment 3: Awesome experiment <a name="experiment_3"></a>
    
Step | Comments
--------- | ------
Hypothesis |  
Set up | 
Results | 
Conclusions |  
Links | [Run], [Report]
    
<p align="right"><a href="#toc">To top</a></p>

## 7. Conclusions <a name="conclusions"></a>
As already mentioned before recognition tasks in music remain complex. Additionally, today as then the community laments a lack of well structured, aligned, large data sets for music information retrieval tasks. In consequence we needed to address the challenge to find an appropriate dataset and model architecture. Few literature for reference was available. We believe that our suggestion is a powerful solution for lyrics recognition. Though, its high computational cost in terms of time and money is evident and implementation remains to be optimized.
<p align="right"><a href="#toc">To top</a></p>

## 8. Next Steps <a name="next_steps"></a>
Further research could be done for:
* melody extraction
* chords transcription
* summary of the lyrics
* pitch recognition
* contribution to larger datasets of high quality
<p align="right"><a href="#toc">To top</a></p>

## 9. References <a name="references"></a>
https://towardsdatascience.com/wav2vec-2-0-a-framework-for-self-supervised-learning-of-speech-representations-7d3728688cae
    
https://ieeexplore.ieee.org/abstract/document/5179014

https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1318.pdf
    
https://ieeexplore.ieee.org/document/6682644?arnumber=6682644
    
https://www.researchgate.net/publication/42386897_Automatic_Recognition_of_Lyrics_in_Singing
    
https://europepmc.org/article/med/20095443
    
https://asmp-eurasipjournals.springeropen.com/articles/10.1155/2010/546047
    
https://arxiv.org/abs/2102.08575
    
https://github.com/facebookresearch/demucs
    
http://ismir2018.ircam.fr/doc/pdfs/35_Paper.pdf
    
https://wandb.ai/site
    
https://cloud.google.com/
    
https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/
    
https://colab.research.google.com/notebooks/intro.ipynb?hl=en
    
https://pytorch.org/docs/stable/torch.html
    
https://transactions.ismir.net/articles/10.5334/tismir.30/
    
https://distill.pub/2017/ctc/
    
https://medium.com/descript/challenges-in-measuring-automatic-transcription-accuracy-f322bf5994f
