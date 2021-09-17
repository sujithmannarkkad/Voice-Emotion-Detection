
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
- If only Inference part need to be checked download the trained model provided and keep in the root foder, and execute the Demo section in the Final-Submission.ipynb after installing and importing necessary libraries.
  
## Acknowledgements

 - [Audio Classification EDA Part 1](https://www.youtube.com/watch?v=mHPpCXqQd7Y)
 - [Audio Classification Part 2](https://www.youtube.com/watch?v=4F-cwOkMdTE&t=970s)
 - [Audio Classification Part 3](https://www.youtube.com/watch?v=uTFU7qThylE)
 - [Audio Classification With Pretrained weights](https://towardsdatascience.com/audio-classification-with-pre-trained-vgg-19-keras-bca55c2a0efe)
 - [Deep Learning Audio Classification](https://medium.com/analytics-vidhya/deep-learning-audio-classification-fcbed546a2dd)
 - [Audio Signal Processing for Machine Learning](https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)


## Contributers

Sujith G (sujithmannarkkad@gmail.com)

  
