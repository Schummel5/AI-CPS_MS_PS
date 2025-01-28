import csv
import requests
import os
from bs4 import BeautifulSoup
import pandas as pd


# This file is for Data Scraping and the Preparation
# We scraped the Data from this URL: https://www.alphaquery.com/stock/AAPL/earnings-history
# The target is to predict what the next EPS of the Apple stock is.

## Step 1:  Data Scraping
URL = "https://www.alphaquery.com/stock/AAPL/earnings-history"
page = requests.get(URL)
directory = "data/"

# Checks if directory exists
os.makedirs(directory, exist_ok=True)



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

#print(df)




