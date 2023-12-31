import os
import sys

path = os.getcwd()
parent_directory = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(parent_directory)

import unittest

#test cases for webscraper

from modules.utils import scrape_news

class TestClass(unittest.TestCase):
    def test_null(self):
        with self.assertRaises(TypeError):
            scrape_news(time_range="10 days ago",selected_topics="",num_articles=2)
    def test_invalidNumber(self):    
        with self.assertRaises(IndexError):
            scrape_news(time_range="10 days ago",selected_topics="Technology news",num_articles=-1)
    def test_type(self):
        with self.assertRaises(TypeError):
            scrape_news(time_range="10 days ago",selected_topics=7,num_articles=2)
    def test_date(self):
        with self.assertRaises(TypeError):
            scrape_news(time_range="why",selected_topics="Technology news",num_articles=2)
#end test cases for webscraper

#TEST CASES FOR SUMMARY FUNCTION
from modules.utils import generate_summaries,ReadingTime,ComplexityLevel
import pandas as pd

class TestClass(unittest.TestCase):
    def test_null(self):
        with self.assertRaises(ValueError):
            df=pd.DataFrame()
            generate_summaries(df,ComplexityLevel.EASY,ReadingTime.ONE_MINUTE)
    def test_invalidNumber(self):    
        with self.assertRaises(ValueError):
            df=pd.DataFrame({
                "Content":[],
                "Test":["test","test"]
            })
            generate_summaries(df,ComplexityLevel.EASY,ReadingTime.ONE_MINUTE)
    def test_type(self):
        with self.assertRaises(ValueError):
            df=scrape_news("10 days ago","world",2)
            temp:pd.DataFrame=df.iloc[:,0:-1]
            outputToSeries=pd.Series([])
            outputToSeries.name="Summaries"
            df=pd.merge(temp,outputToSeries,right_index=True,left_index=True)
            generate_summaries(df,ComplexityLevel.EASY,ReadingTime.ONE_MINUTE)

#END
class TestNewsCraft(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
