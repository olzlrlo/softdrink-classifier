from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
import os

url = "https://www.google.com/search?q=beverage%20can&tbm=isch&tbs=isz:l&rlz=1C5CHFA_enKR968KR974&hl=ko&sa=X&ved=0CAIQpwVqFwoTCPDxtbW_zvMCFQAAAAAdAAAAABAC&biw=1580&bih=939"
driver = webdriver.Chrome("/Users/jieun/Desktop/softdrink_classifier/chromedriver")
driver.get(url)
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver.get(url)

for i in range(500):
    driver.execute_script("window.scrollBy(0, 50000)")

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
image = soup.select('img')

n = 1
imgurl = []

for i in image:
    try:
        imgurl.append(i.attrs["src"])
    except KeyError:
        imgurl.append(i.attrs["data-src"])

for i in imgurl:
    urlretrieve(i, "newdata/" + str(n) + ".jpg")
    n += 1

driver.close()
