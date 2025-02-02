# This file is for Data Scraping and solves the data scraping part of 
# the subgoal 2: Data Scraping and Preperation
# We scraped the Data from this URL: https://www.alphaquery.com/stock/AAPL/earnings-history

# Here are the necessary imports for the data scraping
import csv
import requests
import os
from bs4 import BeautifulSoup
import pandas as pd


## Step 1:  Data Scraping by defining the URL
URL = "https://www.alphaquery.com/stock/AAPL/earnings-history"
page = requests.get(URL)
directory = "data/"

# Checks if the directory data exists
os.makedirs(directory, exist_ok=True)

# This code checks if there is a table on the website (URL) and saves it in a CSV file
soup = BeautifulSoup(page.text, "html.parser")
table = soup.find("table")

if table:
    rows = table.find_all("tr")
    data = []
    for row in rows:
        cells = row.find_all("td")
        if cells:
	        data.append([cell.text.strip() for cell in cells])

    header = [th.text.strip() for th in table.find_all("th")]

    df = pd.DataFrame(data, columns=header)

    csv_file_path = "/tmp/AIBAS_KURS_PS_MS/data/scraped_data.csv"
    df.to_csv(csv_file_path, index=False)

    print(f"The data was successfully stored in the file: {csv_file_path}.")
else:
    print("Keine Tabelle gefunden.")




