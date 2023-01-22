
import unittest
import sys

sys.path.append("../src")

from abstraction import *


class AbstractionUnitTest(unittest.TestCase):
	def test_preflop(self):
		self.assertEqual(get_preflop_cluster_id("AhAd"), get_preflop_cluster_id("AsAc"))
		self.assertEqual(get_preflop_cluster_id("AhAd") + 1, get_preflop_cluster_id("2s2c"))
		self.assertEqual(get_preflop_cluster_id("Ah3d"), 15)
	
	def test_flop(self):
		kmeans_flop, kmeans_turn, kmeans_river = load_kmeans_classifiers()
		self.assertEqual(get_flop_cluster_id(kmeans_flop, "AhAd2s2d3h"), get_flop_cluster_id(kmeans_flop, "AsAd2s2d3h"))
		get_turn_cluster_id(kmeans_turn, "AhAd2s2d3h4d")
		# self.assertEqual(get_turn_cluster_id(kmeans_turn, "AhAd3s3d3h"), get_turn_cluster_id(kmeans_turn, "AsAd3s3d3h"))

if __name__ == '__main__':
	unittest.main()