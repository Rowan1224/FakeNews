import os

import requests

from bs4 import BeautifulSoup


url = 'https://www.alexa.com/siteinfo/bd24live.com'
print(url)
res = requests.get(url)
soup = BeautifulSoup(res.text, 'html.parser')
for soup in soup.find_all(class_='rankmini-rank'):
    rank = str(soup.getText()).replace("\n", "")
    rank = rank.replace("\t", "")
    rank = rank.replace(" ", "")
    rank = rank.replace("#", "")
    rank = rank.replace(",", "")
    print(rank)
    break

