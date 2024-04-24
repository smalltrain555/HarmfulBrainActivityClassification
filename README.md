# Kaggle: Harmful Brain Activity Classification

## About

This resolution repo of [harmful barin activity classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview) for kaggle competition.


## Method

### Data

The input's shape of model is 800x800. 10 minutes' kaggle spectrogram (100x300x4), 10 seconds' eeg spectrogram (100x291x4) and 50 seconds' eeg spectrogram (267x501x4) were got by [SERGEY SAHAROVSKIY's method](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/487110). Then, padding 10 seconds' eeg epectrogram to 100x300x4, and cropping 50 seconds' eeg spectrogram to 200x500x4 which only preserved low frequence singals.

### Preprocess

* Gaussian noise with the probability of 50%
* Horizontal filp with the probability of 50%

Unuseful preprocess methods:
* Mixup
* Vertical filp

### Model

Efficient_b2 was adopted in this competition. Efficient_bx was also used to train, but they didn't get the better result.

Different data were used to train. According to the community disscusion, the data which sum of votes are not less than 10 can improve the performance. So three models were trained based on different votes data: 
* 1.the sum of votes are less than 10
* 2.the sum of votes are not less than 10
* 3.all data.

### Ensemble

In final submission, ensembling models were adopted for increasing scores. Unlike high scores' methods, only were weight average method used in merging different models' results.

## Results

### Single model
  
|Score|Data:Votes>=10|Data:Votes<10|All Data|
|-|-|-|-|
|CV|0.337|0.71|0.602|
|Public|0.366317|0.544724|0.440032|
|Private|0.318288|0.477881|0.385545|

### Ensembling the model trained on a subset with 10 or more votes and the model trained on the remaining data
|Score|0.8/0.2|0.7/0.3|0.5/0.5|
|-|-|-|-|
|Public|0.367039|0.37573|0.399840|
|Private|0.318232|0.324242|0.345662|

*x/y: ensembling model weight of votes>=10 and votes<10*

### Ensembling the model trained on a subset with 10 or more votes and the model trained on the all data
|Score|0.8/0.2|0.7/0.3|0.5/0.5|
|-|-|-|-|
|Public|**0.359052**|0.360623|0.370774|
|Private|**0.310983**|0.311931|0.320502|

*x/y: ensembling model weight of votes>=10 and all data*

## Reference

[Sorry Rapids, not this time](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/487110)

[How To Make Spectrogram from EEG](https://www.kaggle.com/code/cdeotte/how-to-make-spectrogram-from-eeg)
