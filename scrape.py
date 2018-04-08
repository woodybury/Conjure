from bs4 import BeautifulSoup
import requests

url = input("Website: ")

r  = requests.get("http://" +url)

data = r.text

soup = BeautifulSoup(data, "html.parser")

for h1 in soup.find_all('h2'):
    text = h1.text
    print(text)