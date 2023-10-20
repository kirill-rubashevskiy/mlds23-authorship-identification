#!/usr/bin/env python
# coding: utf-8

# парсер v 1.0
# запускается через командную строку с одним аргументом - txt файлом, где на каждой строке указано одно произведение
# требования к парсеру указаны в файле requirements.txt
# создает в папке, где лежит парсер, папку texts c отдельными файлами-произведениями в формате .txt и csv файлом с информацией о найденных и скачанных произведениях; также выводит в командную строку информацию о количестве найденных произведений и названия проиведений, которые не были найдены
# поддерживает продолжение парсинга (когда запускаешь парсер с новым списком произведений)
# отчищает тексты от html тегов, сохраняет структуру абзацев текста
# не сохраняет названия глав
# пока не умеет:
# - находить произведения, названия которых написаны в неправильном регистре
# - отчищать текст сносок автора

import os
import sys
import time
import pandas as pd
from transliterate import translit

# парсинг
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup, SoupStrainer
from selenium.webdriver.chrome.options import Options

SEARCH_URL = 'https://ilibrary.ru/search.phtml?q='

searched = 0
found    = 0
not_found_titles = []

# проверяем, начинаем ли мы парсинг или продолжаем (добавляя новые произведения)
try:
    texts_df = pd.read_csv(os.path.join(os.getcwd(), 'texts_df.csv'), index_col=0)
except:
    texts_df = pd.DataFrame(columns = ['title', 'author', 'file_name'])
    os.mkdir('texts')

def search_title(title):

    global searched
    searched += 1

    driver.get(SEARCH_URL) # заходим на страницу поиска библиотеки
    search_box = driver.find_element(By.NAME, "q") # находим поисковое окошко
    search_box.send_keys(title) # вводим название произведения
    search_box.submit()

    try:
        driver.find_element(By.LINK_TEXT, title).click() # ищем в результатах поиска ссылку на произведение
        global found
        found += 1
        grab_text(title) # делаем парсинг произведения
        return
    except:
        not_found_titles.append(title)
    
    return

def grab_text(title):
    author = driver.find_element(By.CLASS_NAME, 'author').text # сохраняем имя и фамилию автора
    text = grab_page(driver)
    _multipage = driver.find_elements(By.CLASS_NAME, 'navlnktxt') # проверяем, размещено ли произведение на одной или нескольких страницах
    if _multipage:
        _multipage = [0] + _multipage
        while len(_multipage) > 1:
            _multipage[1].click()
            text.extend(grab_page(driver))
            _multipage = driver.find_elements(By.CLASS_NAME, 'navlnktxt')

    # генерируем название файла произведения на основе транслитерированных имени автора и названия произведения
    file_name = '_'.join(translit(f'{author.lower()} {title.lower()}', reversed = True).split())
    
    with open(f'./texts/{file_name}.txt', 'w') as f:
        for line in text:
            f.write(f"{line}\n")

    # сохраняем информацию о произведении в таблицу
    texts_df.loc[len(texts_df.index)] = [title, author, file_name]
    return     

def grab_page(driver):
    src = driver.page_source
    page_soup = BeautifulSoup(src, 'html.parser')
    page_text = page_soup.find_all(attrs={'class': 'p'})
    return [paragraph.get_text() for paragraph in page_text]

# создаем веб драйвер
options = Options()
options.page_load_strategy = 'eager'
driver = webdriver.Chrome(options=options)

# для каждого названия из текстового файла, переданного в качестве аргумента, запускаем парсинг
with open(sys.argv[1]) as search_task:
    for title in search_task:
        title = title.strip()
        print(title)
        search_title(title)

# сохраняем датафрейм
texts_df.to_csv(f'./texts/texts_df.csv')

# выводим на экран информацию о количестве найденных и собранных произведений, а также названия произведений, которые не были найдены
print(f'Found and grabbed {found} title(s) from {searched} searched. Not found titles include:')
for title in not_found_titles:
    print(f' - {title}')