import unittest
from extractor import clean_text

class TestCleanText(unittest.TestCase):
    def test_clean_text(self):
        text = "Hello don't find space computer because remote TV is fast and you're great."
        print clean_text(text)

if __name__ == '__main__':
    unittest.main()