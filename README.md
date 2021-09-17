
# Voice Emotion Detection

This is a voice classification model which was part of a Hacker earth competition.
(https://www.hackerearth.com/challenges/competitive/ia-for-ai/)

We have given with 5815 training audio data set and 2492 validation data set which has 7 different emotions ('neutral','joy','disgust','surprise','sadness','fear','anger').



## DataSet
 Download the below datasets and trained model and unzip and keep in the root folder.
 
- [Train Audio](https://drive.google.com/drive/folders/14bg5LJ45i79m09Y5PgtIPhpjbmcCplq2?usp=sharing)
- [Test Audio](https://drive.google.com/drive/folders/1vPsI74qC6baTBPMl6R6muRMkJlM4fD97?usp=sharing)
- [Final Trained Model Pickle File](https://drive.google.com/file/d/1zD3S6iKsj2Xf7WS2f5DDAC80jTp6Lme3/view?usp=sharing)

## Data Distribution

![App Screenshot](https://raw.githubusercontent.com/sujithmannarkkad/Voice-Emotion-Detection/main/data%20distribution.png)

## Approach

I have extracted Mel-Frequency Cepstral Coefficients(MFCC) from the audio samples using librosa library.
Trained using these features in deep learning and Machine Learning Models and selected best performing model.

Another Approach Tried is by Converting the audio to Image and using CNN, Transfer Learning (VGG19 from imagenet)

I have used full data for training.

## Models Tried
- Custom Deep Learning Model
- CatBoostClassifier
- LogisticRegression
- KNeighborsClassifier
- DecisionTreeClassifier
- GradientBoostingClassifier
- ExtraTreesClassifier
- LGBMClassifierXGBClassifier
- RandomForestClassifier
- OneVsRestClassifier with XGBClassifier


- catboost gave me better result (58.87663) in Hacker Earth while submitting predicion file.

## Results from the models
```bash
CatBoost
Train Accuracy:  1.00,  Test Accuracy:  0.56
Train Precision:  1.00,  Test Precision:  0.58
Train Recall:  1.00,  Test Recall:  0.39


Logreg C=0.01
Train Accuracy:  0.33,  Test Accuracy:  0.30
Train Precision:  0.38,  Test Precision:  0.34
Train Recall:  0.40,  Test Recall:  0.35

KNN
Train Accuracy:  0.61,  Test Accuracy:  0.47
Train Precision:  0.60,  Test Precision:  0.36
Train Recall:  0.46,  Test Recall:  0.32

Decision Tree
Train Accuracy:  1.00,  Test Accuracy:  0.43
Train Precision:  1.00,  Test Precision:  0.36
Train Recall:  1.00,  Test Recall:  0.36

GradientBoost
Train Accuracy:  0.71,  Test Accuracy:  0.52
Train Precision:  0.92,  Test Precision:  0.47
Train Recall:  0.62,  Test Recall:  0.33

ExtraTrees
Train Accuracy:  1.00,  Test Accuracy:  0.58
Train Precision:  1.00,  Test Precision:  0.77
Train Recall:  1.00,  Test Recall:  0.39

LGM
Train Accuracy:  0.98,  Test Accuracy:  0.56
Train Precision:  0.99,  Test Precision:  0.61
Train Recall:  0.98,  Test Recall:  0.38

XGB
Train Accuracy:  0.91,  Test Accuracy:  0.56
Train Precision:  0.97,  Test Precision:  0.63
Train Recall:  0.85,  Test Recall:  0.37

RanFor
Train Accuracy:  0.53,  Test Accuracy:  0.51
Train Precision:  0.41,  Test Precision:  0.25
Train Recall:  0.33,  Test Recall:  0.29

OneVsRestClassifier
Train Accuracy:  0.63,  Test Accuracy:  0.52
Train Precision:  0.87,  Test Precision:  0.50
Train Recall:  0.48,  Test Recall:  0.33
```

## Screenshots

![App Screenshot](https://raw.githubusercontent.com/sujithmannarkkad/Voice-Emotion-Detection/main/image1.png)
![App Screenshot](https://raw.githubusercontent.com/sujithmannarkkad/Voice-Emotion-Detection/main/prediction1.PNG)
![App Screenshot](https://raw.githubusercontent.com/sujithmannarkkad/Voice-Emotion-Detection/main/prediction2.png)

  
# Installation and Execution
## Requirements

- Python 3.5+
- catboost
- librosa
- Gradio ( for quick demo)
- GPU enabled machine ( this is needed for quick execution, you can use google colab)

## Set Up

```bash
!pip install catboost
!pip install librosa
!pip install mutagen
!pip install gradio
```

- Download the train, test files using the given link. Unzip to the root folder.
- If only Inference part need to be checked, download the trained model provided and keep in the root foder, and execute the Demo section in the Final-Submission.ipynb after installing and importing necessary libraries.
- Use GPU enabled instance for faster execution.
- Read the instructions carefully in each cells before executing them.
- Execution of couple of cells may take more than 1 hr, alternative way for that is to skip the cells and read from the pickled files. The instructions for the same is given in the notebook.
- Code for auto download of submission file will work only in colab (enable allow multiple files download option in browser when it is prompted). 
  
## Acknowledgements

 - [Audio Classification EDA Part 1](https://www.youtube.com/watch?v=mHPpCXqQd7Y)
 - [Audio Classification Part 2](https://www.youtube.com/watch?v=4F-cwOkMdTE&t=970s)
 - [Audio Classification Part 3](https://www.youtube.com/watch?v=uTFU7qThylE)
 - [Audio Classification With Pretrained weights](https://towardsdatascience.com/audio-classification-with-pre-trained-vgg-19-keras-bca55c2a0efe)
 - [Deep Learning Audio Classification](https://medium.com/analytics-vidhya/deep-learning-audio-classification-fcbed546a2dd)
 - [Audio Signal Processing for Machine Learning](https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)


## Contributers

Sujith G (sujithmannarkkad@gmail.com)

  
