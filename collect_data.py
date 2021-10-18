import os
import re
from astropy.io import fits
import numpy as np
from typing import Iterable


class Expo:
    """Holds flats and darks for same exposition"""
    def __init__(self, expo: float, flats: Iterable[np.ndarray], darks: Iterable[np.ndarray]):
        if flats is None:
            raise ValueError('Exposition %f has no flats' % expo)
        if darks is None:
            raise ValueError('Exposition %f has no darks' % expo)
        self.expo = expo
        self.flats = flats
        self.darks = darks

    def apply_indices(self, indices: tuple[tuple[int]]) -> np.ndarray:
        """Applies indices, and calculates 'lights' = <flats> - <darks> and errors"""
        # apply indices and transform list of arrays in 2-d array, where ich row is one experiment
        f_cut = np.array(list(map(lambda el: el[indices], self.flats)))
        d_cut = np.array(list(map(lambda el: el[indices], self.darks)))

        if f_cut.ndim > 1:
            f_mean = np.mean(f_cut, 0)
            d_f = np.std(f_cut, 0)
        else:
            f_mean = f_cut
            d_f = np.zeros_like(f_mean)

        if d_cut.ndim > 1:
            d_mean = np.mean(d_cut, 0)
            d_d = np.std(d_cut, 0)
        else:
            d_mean = d_cut
            d_d = np.zeros_like(d_mean)

        return f_mean - d_mean, d_f + d_d


def scan_folder(folder: str, recursive = False) -> tuple[list[str], list[str]]:
    """
    Scans given folder and returns list of darks and flats files
    :param folder: Folder to scan
    :param recursive: Is scan subdirectories
    :return: Two tuples
        darks: Tuple of filenames for darks
        flats: Tuple of filenames for flats
    """
    if not os.path.isdir(folder):
        raise AttributeError('Argument "folder" must be a directory')
    contents = os.listdir(folder)
    darks = []
    flats = []
    for file in contents:
        path = os.path.join(folder, file)
        if os.path.isdir(path):
            if recursive:
                d, f = scan_folder(path, True)
                darks += d
                flats += f
        elif re.match(r"^.*dark.*\.fts$", path):
            darks.append(path)
        elif re.match(r"^.*flat.*\.fts$", path):
            flats.append(path)

    return (darks, flats)

def group_by_expo(darks: Iterable[str], flats: Iterable[str]) -> tuple[Expo]:
    """
    Groups images by exposition
    :param darks: List of darks filenames
    :param flats: List of flats filenames
    :return: List of Expo's
    """
    exp_2_darks = {}
    exp_2_flats = {}

    _group_by_expo(exp_2_darks, darks)
    _group_by_expo(exp_2_flats, flats)

    expos = []
    for exp in set(tuple(exp_2_darks.keys()) + tuple(exp_2_flats.keys())):
        e = Expo(
            exp,
            exp_2_flats.get(exp),
            exp_2_darks.get(exp)
        )
        expos.append(e)
    return tuple(expos)


def process_folder(folder: str, recursive = False) -> tuple[Expo]:
    """Applies scan_folder and group_by_expo"""
    darks, flats = scan_folder(folder, recursive)
    expos = group_by_expo(darks, flats)
    return expos


def _group_by_expo(container: dict, files: Iterable[str]):
    for file in files:
        hdul = fits.open(file)
        if hdul[0].header['EXPTIME'] not in container.keys():
            container[hdul[0].header['EXPTIME']] = []
        container[hdul[0].header['EXPTIME']].append(hdul[0].data)
        hdul.close()
