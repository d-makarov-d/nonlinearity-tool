import os
import unittest
from shutil import rmtree

from collect_data import scan_folder
from nonlinearity import process_area_file


class Files(unittest.TestCase):
    def setUp(self) -> None:
        """Make test folders"""
        try:
            os.mkdir('test_tmp')
            for i in range(10):
                open('test_tmp/dark%i.fts' % i, 'w').close()
            os.mkdir('test_tmp/f1')
            open('test_tmp/f1/f1_00.txt', 'w').close()
            open('test_tmp/f1/f1_00', 'w').close()
            for i in range(10):
                open('test_tmp/f1/f1_flat_%i.fts' % i, 'w').close()
            os.mkdir('test_tmp/f1/f2')
            for i in range(5):
                open('test_tmp/f1/f2/f2_flat_%i.fts' % i, 'w').close()
            for i in range(5):
                open('test_tmp/f1/f2/f2_dark_%i.fts' % i, 'w').close()
        except FileExistsError:
            pass

    def test_read_folder_rec(self):
        darks, flats = scan_folder('test_tmp', True)
        self.assertEqual(len(darks), 15)
        self.assertEqual(len(flats), 15)

    def test_process_area_file(self):
        indices = process_area_file('tests/test_area.txt')
        expected = (
            (56,1), (56,2), (56,3), (56,4), (56,5), (56,6), (56,7), (56,8), (56,9), (56,10), (56,11), (56,18),
            (74,1), (74,2),
            (65, 1), (65, 2), (65, 5)
        )
        expected = (
            12 * (56, ) + 2 * (74, ) + 3 * (65, ),
            tuple(range(1, 12)) + (18, 1, 2, 1, 2, 5)
        )
        self.assertEqual(indices, expected)

    def tearDown(self) -> None:
        """Clear test data"""
        rmtree('test_tmp')

