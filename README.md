# Phishing Detection Using TCNs (Temporal Neural Network) 
This research was inspired by the workings of [this following paper](https://arxiv.org/pdf/1803.01271.pdf) and is the driving force to how this project will continue.

## What are TCNs?
* TCNs stand for Temporal Convolutional Networks. They are distinguished by their dialated causal 1D convolutions that allow them to hold more mememroy than network arictectures such as RNNs or their variants (GRU, LSTM).
* They are also special in their receptive field that describes how far the model can see into the past.
* It is provided as a great solution to combat the issue of RNNs and vanashing gradients while not being as computionally expensive as LSTMs.
  

## The Framework of this Project:
While TCNs are most noteably used in sequence modeling, this project is part of my thesis work for investigating whether TCNs can uphold the level of effectiveness when it comes to detecting malicous URLs used in phishing schemes.
The results of this research will be found by testing the finished model on several datasets that contain maclious and benign URLs. The model will be also measured on severla evalutaion metrics which include 
1. Accuracy
2. Precision
3. Recall and
4. F-1 score.

Of course, the goals of this experiment is to obtain the most desirable number for each of these metrics while also comparing it to other research that is already present in the literature in order to really undertsnad how the overall performance of the model is like.
