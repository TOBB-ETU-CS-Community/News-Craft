import os
import sys

path = os.getcwd()
parent_directory = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(parent_directory)

import unittest

#test cases for webscraper

from modules import utils

class TestClass(unittest.TestCase):
    def test_null(self):
        with self.assertRaises(TypeError):
            utils.scrape_news(time_range="10 days ago",selected_topics="",num_articles=2)
    def test_invalidNumber(self):    
        with self.assertRaises(IndexError):
            utils.scrape_news(time_range="10 days ago",selected_topics="Technology news",num_articles=-1)
    def test_type(self):
        with self.assertRaises(TypeError):
            utils.scrape_news(time_range="10 days ago",selected_topics=7,num_articles=2)
    def test_date(self):
        with self.assertRaises(TypeError):
            utils.scrape_news(time_range="why",selected_topics="Technology news",num_articles=2)
#end test cases for webscraper

class TestNewsCraft(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
