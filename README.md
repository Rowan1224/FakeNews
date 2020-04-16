## BanFakeNews: A Dataset for Detecting Fake News in Bangla

## Fake News Dataset can be collected from [here](https://drive.google.com/uc?export=download&id=1DTozpGosyTo6ZIguaqgrI9BlVdyUAiZI)

- ## Authentic-48K.csv and Fake-1K.csv conatain following coloumns:

   - articleID : ID of the news
   - domain : News publisher's site name
   - date : Published Date
   - category: Category of the news
   - headline: Headline of the news
   - content: Article or body of the news
   - label: 1 or 0 . '1' for authentic '0' for fake

- ## LabeledAuthentic-7K.csv and LabeledFake-1K.csv conatain following coloumns:

   - articleID : ID of the news
   - domain : News publisher's site name
   - date : Published Date
   - category: Category of the news
   - source: Source of the news. (One who can verify the claim of the news)
   - relation: Related or Unrelated. Related if headline matches with content's claim otherwise it is labeled as Unrelated
   - headline: Headline of the news
   - content: Article or body of the news
   - label: 1 or 0 . '1' for authentic '0' for fake
   - F-type: Type of fake news (Clickbait, Satire, Fake(Misleading or False Context)) [This is present on LabeledFake-1K.csv only]



## Soon all the codes for experiments will be available in this repository. 
