from bs4 import BeautifulSoup
import requests
import os
import pandas as pd
import pdb
from tqdm import tqdm
baseurl = "https://papers.nips.cc"
url_2022 = baseurl + "/paper_files/paper/2022"
print(url_2022)
url_2022_response = requests.get(url_2022)
soup_2022 = BeautifulSoup(url_2022_response.content, "html.parser")
papers_2022 = soup_2022.find_all("li", {"class": "conference"})
papers_2022 = [(x.a.string, x.a["href"]) for x in papers_2022]

title = []
authors = []
abstract = []
year = []

bad_pages = []
for paper in tqdm(papers_2022):
    paper_page_url = baseurl+paper[1]
    paper_page_url_response = requests.get(paper_page_url)
    paper_soup = BeautifulSoup(paper_page_url_response.content, "html.parser")
    paper_p = paper_soup.find_all("p")
    paper_title = paper[0]
    paper_author = paper_p[1].i.string
    if paper_p[2].string is not None:
        paper_abstract = paper_p[2].string
    elif paper_p[2].p is not None and paper_p[2].p.string is not None:
        paper_abstract = paper_p[2].p.string
    else:
        bad_pages.append(paper_author)
        continue

    title.append(paper_title)
    authors.append(paper_author)
    abstract.append(paper_abstract)
    year.append("2022")

df = pd.DataFrame(data={"title":title, "authors":authors, "abstract": abstract, "year": year})
df.to_csv("nips_2022.csv", index=False)
print(bad_pages)
print(df.head())