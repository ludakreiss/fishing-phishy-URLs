

# Website URL Phishing Detection System using Machine Learning Techniques

This is a research done to explore how well machine learning models can perfom as phishing detection systems and whether different models can outperofrm other models in such task. 

## Table of Contents:
* [Purpose of Project](#item-one)
* [Working Tree of the Repository](#item-two)
* [Links to the Datasets Used](#item-three)
* [Metrics Used to Evaluate Each Model](#item-six)
* [Results of Each Dataset After Testing](#item-seven)
* [Github Repositories](#item-four)
* [fishing-phishy-URLs Website](#item-five)


<a id="item-one"></a>
## Purpose of Project
The idea to this project started with observing how much of phishing continues to be an ongoing problem in the world of cybersecuirty and cybercrimes. Thus, the thought experiment of applying mahcine learning methods in phishing detection system has turned into reality. 

In this project, 3 datasets have been expeirmented on using 5 different models. In addition, the project showcases how factors such as imbalanced vs balanced datastes and the types of fetaures extracted may affect the effeciency of a model.  

<a id="item-two"></a>
## Working Tree of the Repository
```
.devcontainer
   |-- devcontainer.json
.gitignore
.streamlit
   |-- config.toml
LICENSE
Models
   |-- Dataset #1
   |   |-- Linear SVC
   |   |   |-- Linear SVC.ipynb
   |   |   |-- LinearSVC #1.joblib
   |   |-- MLP
   |   |   |-- MLP #1.joblib
   |   |   |-- MLP.ipynb
   |   |-- Metrics of Dataset #1.csv
   |   |-- Random Forest
   |   |   |-- Random Forest.ipynb
   |   |   |-- rand_forest #1.joblib
   |   |-- TCN
   |   |   |-- TCN #1.h5
   |   |   |-- TCN.ipynb
   |   |-- XGBoost
   |   |   |-- XGBoost #1.joblib
   |   |   |-- XGBoost.ipynb
   |-- Dataset #2
   |   |-- Linear SVC
   |   |   |-- Linear SVC.ipynb
   |   |   |-- LinearSVC #2.joblib
   |   |   |-- scaler.joblib
   |   |-- MLP
   |   |   |-- MLP #2.joblib
   |   |   |-- MLP.ipynb
   |   |   |-- scaler.joblib
   |   |-- Metrics of Dataset #2.csv
   |   |-- Random Forest
   |   |   |-- Random Forest.ipynb
   |   |   |-- rand_forest #2.joblib
   |   |-- TCN
   |   |   |-- TCN #2.h5
   |   |   |-- TCN #2.weights.h5
   |   |   |-- TCN.ipynb
   |   |   |-- scaler.joblib
   |   |   |-- tcn_model_architecture.json
   |   |-- XGBoost
   |   |   |-- XGBoost #2.joblib
   |   |   |-- XGBoost.ipynb
   |-- Dataset #3 (Oversampling)
   |   |-- Linear SVC
   |   |   |-- Linear SVC.ipynb
   |   |   |-- LinearSVC #3.joblib
   |   |-- MLP
   |   |   |-- MLP #3.joblib
   |   |   |-- MLP.ipynb
   |   |-- Metrics of Dataset #3 (Oversampling).csv
   |   |-- Random Forest
   |   |   |-- Random Forest.ipynb
   |   |   |-- rand_forest #3.joblib
   |   |-- TCN
   |   |   |-- TCN #3.h5
   |   |   |-- TCN.ipynb
   |   |-- XGBoost
   |   |   |-- XGBoost #3.joblib
   |   |   |-- XGBoost.ipynb
   |-- Dataset #3 (Undersampling)
   |   |-- Linear SVC
   |   |   |-- Linear SVC.ipynb
   |   |   |-- LinearSVC (Undersampling) #3.joblib
   |   |-- MLP
   |   |   |-- MLP (Undersampling)#3.joblib
   |   |   |-- MLP.ipynb
   |   |-- Metrics of Dataset (Undersampling)#3.csv
   |   |-- Random Forest
   |   |   |-- Random Forest.ipynb
   |   |   |-- rand_forest (Undersampling) #3.joblib
   |   |-- TCN
   |   |   |-- TCN (Undersampling)#3.h5
   |   |   |-- TCN.ipynb
   |   |-- XGBoost
   |   |   |-- XGBoost(Undersampling) #3.joblib
   |   |   |-- XGBoost.ipynb
Processed Data
   |-- Dataset #1 After Feature Extraction.csv
   |-- Dataset#2 After Feature Extraction.csv
   |-- Dataset#3 After Feature Extraction.csv
   |-- Datasets.ipynb
   |-- shortening_services.txt
README.md
Raw Data
   |-- Dataset #1.csv
   |-- Dataset #2.csv
   |-- Dataset #3.csv
Split Data
   |-- Split Dataset #1.ipynb
   |-- Split Dataset #2.ipynb
   |-- Split Dataset #3 (Oversample).ipynb
   |-- Split Dataset #3 (Undersample).ipynb
   |-- Testing Dataset
   |   |-- X_test #1.csv
   |   |-- X_test #2.csv
   |   |-- X_test #3.csv
   |   |-- X_test (Undersample)#3.csv
   |   |-- y_test #1.csv
   |   |-- y_test #2.csv
   |   |-- y_test #3.csv
   |   |-- y_test (Undersample)#3.csv
   |-- Training Dataset
   |   |-- X_train #1.csv
   |   |-- X_train #2.csv
   |   |-- X_train #3.csv
   |   |-- X_train (Undersample)#3.csv
   |   |-- y_train #1.csv
   |   |-- y_train #2.csv
   |   |-- y_train #3.csv
   |   |-- y_train (Undersample)#3.csv
   |-- Validation Dataset
   |   |-- X_val #1.csv
   |   |-- X_val #2.csv
   |   |-- X_val #3.csv
   |   |-- X_val (Undersample)#3.csv
   |   |-- y_val #1.csv
   |   |-- y_val #2.csv
   |   |-- y_val #3.csv
   |   |-- y_val (Undersample)#3.csv
TCN_build.py
feature_extractor.py
fishing-phishy-URLs.py
requirements.txt
runtime.txt
```

<a id="item-three"></a>
## Links to the Datasets Used
* [Dataset #1](https://data.mendeley.com/datasets/vfszbj9b36/1)
* [Dataset #2](https://data.mendeley.com/datasets/c2gw7fy2j4/3)
* [Dataset #3](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset/data)

<a id="item-six"></a>
## Metrics Used to Evaluate Each Model
* Accuracy
* Precision 
* Recall
* F1-Score

<a id="item-seven"></a>
## Results of Each Dataset After Testing
### Dataset #1
| Model           | Accuracy | Precision | Recall | F1_score |
|-----------------|----------|-----------|--------|----------|
| LinearSVC       | 0.7967   | 0.8402    | 0.1530 | 0.2589   |
| Random Forest   | 0.8703   | 0.8723    | 0.5165 | 0.6488   |
| XGBoost         | 0.8663   | 0.8763    | 0.4934 | 0.6314   |
| MLP             | 0.8690   | 0.8758    | 0.5073 | 0.6424   |
| TCN             | 0.8691   | 0.8509    | 0.5282 | 0.6518   |

### Dataset #2

| Model           | Accuracy | Precision | Recall | F1_score |
|-----------------|----------|-----------|--------|----------|
| LinearSVC       | 0.8694   | 0.8592    | 0.8923 | 0.8754   |
| Random Forest   | 0.9009   | 0.9140    | 0.8912 | 0.9024   |
| XGBoost         | 0.8950   | 0.8926    | 0.9048 | 0.8986   |
| MLP             | 0.8962   | 0.8937    | 0.9059 | 0.8998   |
| TCN             | 0.8997   | 0.9138    | 0.8889 | 0.9011   |


### Dataset #3 (Oversampled)
| Model           | Accuracy | Precision | Recall | F1_score |
|-----------------|----------|-----------|--------|----------|
| LinearSVC       | 0.6253   | 0.3041    | 0.8379 | 0.4463   |
| Random Forest   | 0.7301   | 0.3866    | 0.8484 | 0.5312   |
| XGBoost         | 0.7312   | 0.3878    | 0.8493 | 0.5325   |
| MLP             | 0.7304   | 0.3870    | 0.8488 | 0.5316   |
| TCN             | 0.7343   | 0.3875    | 0.8166 | 0.5256   |

### Dataset #3 (Undersampled)
| Model           | Accuracy | Precision | Recall | F1_score |
|-----------------|----------|-----------|--------|----------|
| LinearSVC       | 0.6234   | 0.3036    | 0.8425 | 0.4464   |
| Random Forest   | 0.7333   | 0.3902    | 0.8532 | 0.5355   |
| XGBoost         | 0.7308   | 0.3880    | 0.8550 | 0.5338   |
| MLP             | 0.7320   | 0.3892    | 0.8556 | 0.5351   |
| TCN             | 0.7334   | 0.3879    | 0.8291 | 0.5285   |


<a id="item-four"></a>
## Github Repositories
Some Github Repoistories that have contributed greatly in this project inlcude:
* [keras-tcn python module](https://github.com/philipperemy/keras-tcn)
* [list of URL shortners ](https://github.com/PeterDaveHello/url-shorteners.git)

<a id="item-five"></a>
## fishing-phishy-URLs Website
This project has been sucessfully deployed using [Streamlit](https://streamlit.io/cloud). To test out whether a URL is phishing or not on different types of models, check the link to the [website](https://fishing-phishy-urls.streamlit.app/)!

