
import unittest
import os
import sys
import shutil
import treys
from tqdm import tqdm


if __name__ == "__main__":
	devpath = os.path.relpath(os.path.join('..'), start=os.path.dirname(__file__))
	sys.path = [devpath] + sys.path

class UnitTests(unittest.TestCase):
	def test_1(self):
		self.assertEqual(1, 1)
	
if __name__ == "__main__":
	unittest.main()