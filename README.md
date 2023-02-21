# Article-MultiClass-Text-Classification-DeepLearning

While manually classifying articles to their respective categories takes a lot of time and effort, this neural network model provides an automated and efficient solution that can handle large volumes of data quickly and accurately. The goal of this project is to classify articles into one of five categories: sports, tech, business, entertainment, or politics.<br/>
The F1 Accuracy Score achieved is 93%<br/>
Hopefully, this project can be useful for various applications, such as news recommendation systems, content moderation, and trend analysis. This README will demonstrate more details on the dataset, methodology, and results of the project. 

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Dataset
Source of Data: <br/> https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv<br/>
I've loaded the dataset by using the Pandas library. <br/>
`pd.read_csv(URL)`

## Data Cleaning
For the data cleaning, I've used the Regex library to remove:
* Numbers and Punctuations
* URLs
* Contractions

The dataset did not contain single quotes for contractions. Hence it displayed like this *'spain s'*. Due to that, I've removed anything with a single letter from the texts. After removing the outliers, I transformed them into lowercase

## Data Preprocessing
* Data Tokenization
The tokenization of the text data was done by using the Tokenizer class from Tensorflow. the tokenizer was trained on the text data which allows to learn the vocabulary of the text and assign unique indexes to each word. Then the sentences were transformed into sequences of intergers.

* Padding and Truncating
To standardize the length of the sentences, I did post padding and truncating using the `pad_sequence()` method.

* One-Hot-Encoding on the 5 categories
The target variable which are the categories were encoded using the `OneHotEncoder()` 

* Train-Test-Split
The data was then split into training and test sets.<br/>
  - Train size: 0.7
  - Test_size: 0.3

## Model Development

<p align="center">
  <img src="https://github.com/natashanazamil/Article-MultiClass-Text-Classification-DeepLearning/blob/main/images/model_architecture.png?raw=true" alt="Model Architecture">
  <br>
  <em>Model Architecture</em>
</p>

The model was compiled and trained using:
* loss: categorical_crossentropy
* optimizer: adam
* metrics: accuracy
* epochs: 100
* callbacks: TensorBoard, EarlyStopping

<p align="center">
  <img src="https://github.com/natashanazamil/Article-MultiClass-Text-Classification-DeepLearning/blob/main/images/model_summary.PNG?raw=true" alt="Model Summary">
  <br>
  <em>Model Summary</em>
</p>

## Model Analysis
The model analysis was done using the confusion matrix and classification report from  `sklearn.metrics`

### TensorBoard
<p align="center">
  <img src="https://github.com/natashanazamil/Article-MultiClass-Text-Classification-DeepLearning/blob/main/images/tensorboard_acc.PNG?raw=true" alt="Tensorboard - Training vs Validation">
  <br>
  <em>Tensorboard - Training vs Validation</em>
</p>

The graph shows that it did not overfit. Although it did not converge, the gap between the training and validation points were not far.

### Confusion Matrix
<p align="center">
  <img src="https://github.com/natashanazamil/Article-MultiClass-Text-Classification-DeepLearning/blob/main/images/confusion_matrix.PNG?raw=true" alt="Confusion Matrix">
  <br>
  <em>Confusion Matrix</em>
</p>

From the confusion matrix, we can see that most of the texts were classified accurately.

### Classification Report
<p align="center">
  <img src="https://github.com/natashanazamil/Article-MultiClass-Text-Classification-DeepLearning/blob/main/images/classification_report.PNG?raw=true" alt="Classification Report">
  <br>
  <em>Classification Report</em>
</p>

As for the classification report, we can see that the model managed to achieve 93% of the F1 Accuracy Score.

## Model Deployment
The model was then deployed as .h5 file. As well as the tokenizer and one-hot-encoder as .json and .pkl file respectively. They can be found in the **model** file.
