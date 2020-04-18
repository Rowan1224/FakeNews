## BanFakeNews: A Dataset for Detecting Fake News in Bangla (LREC 2020)


`Observing the damages that can be done by the rapid propagation of fake news in various sectors like politics and finance, automatic identification of fake news using linguistic analysis has drawn the attention of the research community. However, such methods are largely being developed for English where low resource languages remain out of the focus. But the risks spawned by fake and manipulative news are not confined by languages. In this work, we propose an annotated dataset of ~50K news that can be used for building automated fake news detection systems for a low resource language like Bangla. Additionally, we provide an analysis of the dataset and develop a benchmark system with state of the art NLP techniques to identify Bangla fake news. To create this system, we explore traditional linguistic features and neural network based methods.
We expect this dataset will be a valuable resource for building technologies to prevent the spreading of fake news and contribute in research with low resource languages.`

**Authors**
* Md Zobaer Hossain<sup>1</sup>
* Md Ashraful Rahman<sup>1</sup>
* Md Saiful Islam<sup>1</sup>
* Sudipta Kar<sup>2</sup>
<sup>1</sup> Shahjalal University of Science and Technology
<sup>2</sup> University of Houston


## BanFakeNews dataset is available [here](https://drive.google.com/uc?export=download&id=1DTozpGosyTo6ZIguaqgrI9BlVdyUAiZI).

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
| date | Category of the news|
| category | Category of the news|
| headline | Headline of the news|
| content | Article or body of the news|
| label | 1 or 0 . '1' for authentic '0' for fake|

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


## Bibtex
```
@InProceedings{Hossain20.1084,
 author = {Md Zobaer Hossain, Md Ashraful Rahman, Md Saiful Islam, Sudipta Kar},
 title = "{BanFakeNews: A Dataset for Detecting Fake News in Bangla}",
 booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2020)},
 year = {2020},
 publisher = {European Language Resources Association (ELRA)},
language = {english}
}
```

## Source code will be available soon.
