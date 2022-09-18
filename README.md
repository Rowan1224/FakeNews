## BanFakeNews: A Dataset for Detecting Fake News in Bangla (LREC 2020)


**Recent Update**


A new notebook (``BanFake.ipynb``) has been added to the repository to make it easier to use Huggingface’s Transformer package and experiment with new language models. 


**Abstract**

Observing the damages that can be done by the rapid propagation of fake news in various sectors like politics and finance, automatic identification of fake news using linguistic analysis has drawn the attention of the research community. However, such methods are largely being developed for English where low resource languages remain out of the focus. But the risks spawned by fake and manipulative news are not confined by languages. In this work, we propose an annotated dataset of ~50K news that can be used for building automated fake news detection systems for a low resource language like Bangla. Additionally, we provide an analysis of the dataset and develop a benchmark system with state of the art NLP techniques to identify Bangla fake news. To create this system, we explore traditional linguistic features and neural network based methods.
We expect this dataset will be a valuable resource for building technologies to prevent the spreading of fake news and contribute in research with low resource languages.


**Authors**
* Md Zobaer Hossain<sup>1</sup>
* Md Ashraful Rahman<sup>1</sup>
* Md Saiful Islam<sup>1</sup>
* Sudipta Kar<sup>2</sup>

<sup>1</sup> Shahjalal University of Science and Technology

<sup>2</sup> University of Houston


## BanFakeNews dataset is available [here](https://www.kaggle.com/cryptexcode/banfakenews).

#### List of files
* Authentic-48K.csv
* Fake-1K.csv
* LabeledAuthentic-7K.csv
* LabeledFake-1K.csv

**File Format**
Authentic-48K.csv and Fake-1K.csv

| Column Title   | Description |
| ------------- |------------- |
| articleID      | ID of the news |
| domain      | News publisher's site name      |
| date | Published Date|
| category | Category of the news|
| headline | Headline of the news|
| content | Article or body of the news|
| label | 1 or 0 . '1' for authentic '0' for fake|

LabeledAuthentic-7K.csv, LabeledFake-1K.csv

|Column Title   |Description |
|------------- |------------- |
| articleID | ID of the news |
| domain | News publisher's site name |
| date | Published Date |
| category | Category of the news |
| source | Source of the news. (One who can verify the claim of the news) |
| relation | Related or Unrelated. Related if headline matches with content's claim otherwise it is labeled as Unrelated |
| headline | Headline of the news |
| content | Article or body of the news |
| label | 1 or 0 . '1' for authentic '0' for fake |
| F-type | Type of fake news (Clickbait, Satire, Fake(Misleading or False Context))

**F-type** is only present in LabeledFake-1K.csv


## INSTALLATION
 Requires the following packages:
 * Python 3.7

It is recommended to use virtual environment packages such as **virtualenv** or **conda** 
Follow the steps below to setup project:
* Clone this repository. `git clone https://github.com/Rowan1697/FakeNews.git`
* Use this command to install required packages `pip install -r requirements.txt`
* Run the setup.sh file to download additional data and setup pre-processing

## Usage
1. Download Fake News data from [here](https://www.kaggle.com/cryptexcode/banfakenews).
2. Unzip the folder
3. Ensure the folder name is "Fake News Dataset"

**Basic Experiments**
* Go to Models/Basic folder
* Use **python n-gram.py [Experiment Name] [Model] [-s](optional)** to run an experiment. For example: `python n-gram.py Emb_F SVM -s` will run the Emb_F experiment using SVM Model. Use -s to Save the results. 
* **Experiment Names** (Please follow the paper to read the details about experiments) : 
    * Unigram
    * Bigram
    * Trigram
    * U+B+T
    * C3-gram
    * C4-gram
    * C5-gram
    * C3+C4+C5
    * Lexical
    * POS
    * L_POS
    * Emb_F
    * Emb_N
    * L+POS+E_F
    * L+POS+E_N
    * MP
    * L+POS+E_F+MP
    * L+POS+E_N+MP
    * all_features
* Models:
    * SVM (Support Vector Machine)
    * LR (Logistic Regression)
    * RF (Random Forest)

**NN Experiments**
* Go to Models/NN folder
* Use **python main.py [Model] [-g](optional)** to run an experiment. For example: `python main.py CNN -g` will run the experiment using CNN model. Use -g to run in GPU.
* Models:
    *  CNN
    *  LSTM

**BERT**
* Go to Models/BERT folder
* Use **python bert.py [epoch]** to run an experiment. For example: `python bert.py 3` will run an experiment with 3 epochs

## Bibtex
```
@InProceedings{Hossain20.1084,
 author = {Md Zobaer Hossain, Md Ashraful Rahman, Md Saiful Islam, Sudipta Kar},
 title = "{BanFakeNews: A Dataset for Detecting Fake News in Bangla}",
 booktitle = {Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC 2020)},
 year = {2020},
 publisher = {European Language Resources Association (ELRA)},
language = {english}
}
```
