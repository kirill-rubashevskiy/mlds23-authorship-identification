"""
File contains functions and classes for the text parser's operation.
"""

import pandas as pd
from typing import Union, Any, Optional
from transliterate import translit

# parsing
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup, SoupStrainer
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException


class TextObject:

    def __init__(self,
                 title: str,
                 author: str,
                 author_id: int,
                 file_name: str,
                 text_link: str,
                 text: list[str]):
        self.title = title
        self.author = author
        self.author_id = author_id
        self.file_name = file_name
        self.text_link = text_link
        self.text = text


class IlibParser:

    def __init__(self,
                 new_search: bool = True):

        options = Options()
        options.page_load_strategy = 'eager'

        self.driver = webdriver.Chrome(options=options)
        self.search_url = 'https://ilibrary.ru/search.phtml?q='
        self.searched = []
        self.found = []
        self.not_found = []
        if new_search:
            self.found_df = pd.DataFrame(columns=['title', 'author', 'author_id', 'file_name', 'text_url'])
        else:
            self.found_df = pd.read_csv('./parsed_data/found_df.csv', index_col=0)

    def _grab_page(self) -> str:

        src = self.driver.page_source
        page_soup = BeautifulSoup(src, 'html.parser', parse_only=SoupStrainer("span", {"class": "p"}))

        return page_soup.text

    def _grab_text(self, title: str) -> tuple[list[Union[str, Any]], str]:

        text_url = self.driver.current_url
        author = self.driver.find_element(By.CLASS_NAME, 'author').text
        if author in self.found_df.author.unique():
            author_id = self.found_df[self.found_df.author == author].values[0,2]
        else:
            author_id = self.found_df.author.nunique()

        text = self._grab_page()

        _multipage = self.driver.find_elements(By.CLASS_NAME,
                                               'navlnktxt')
        if _multipage:
            _multipage = [0] + _multipage
            while len(_multipage) > 1:
                _multipage[1].click()
                text += self._grab_page()
                _multipage = self.driver.find_elements(By.CLASS_NAME, 'navlnktxt')

        # generate file name to store text
        file_name = '_'.join(translit(f'{author.lower()} {title.lower()}', reversed=True).split())

        text_row = [title, author, author_id, file_name, text_url]

        if author_id < 10:
            text = f'author_id_0{str(author_id)}' + text
        else:
            text = f'author_id_{str(author_id)}' + text

        return text_row, text

    def search_title(self,
                     title: str) -> Optional[TextObject]:

        if title in self.searched:
            return

        if title in self.found_df.title.values.tolist():
            self.searched.append(title)
            self.found.append(title)

            return

        # search for title
        self.searched.append(title)
        self.driver.get(self.search_url)
        search_box = self.driver.find_element(By.NAME, "q")
        search_box.send_keys(title)
        search_box.submit()

        try:
            text_link = self.driver.find_element(By.LINK_TEXT, title)

        except NoSuchElementException:
            self.not_found.append(title)
            return

        text_link.click()
        text_row, text = self._grab_text(title)
        self.found_df.loc[len(self.found_df.index)] = text_row
        self.found.append(title)
        text_props = text_row + [text]

        return TextObject(*text_props)

    def end_search(self):
        self.driver.close()
