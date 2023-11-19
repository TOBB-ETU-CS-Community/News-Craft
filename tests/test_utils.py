import os
import sys

path = os.getcwd()
parent_directory = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(parent_directory)

import unittest


class TestNewsCraft(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
