import argparse
import re
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt

from collect_data import process_folder, Expo


class Area:
    """Holds number of pixels for different expositions"""
    def __init__(self, expos: Iterable[Expo], indices: tuple[tuple[int]]):
        exp_2_light = dict(map(
            lambda e: (e.expo, e.apply_indices(indices)),
            expos
        ))
        # sort by expositions (aka sort by key)
        self.raw = dict(sorted(exp_2_light.items()))

        # unpack raw values
        self.expos = np.array(tuple(self.raw.keys()))
        self.pix = np.array(tuple(map(lambda el: el[0], self.raw.values())))
        d_pix = np.array(tuple(map(lambda el: el[1], self.raw.values())))
        # average signal for the whole area for each exposition
        self.avg_pix = np.mean(self.pix, 1)
        self.d_avg_pix = np.mean(d_pix, 1) + np.std(self.pix, 1)
        # fit the average sig(exp) by least square polynomial fit
        self.a, self.b = np.polyfit(self.expos, self.avg_pix, 1)
        self.nonlin = self.avg_pix - self.lin_sig(self.expos)

    def lin_sig(self, exp) -> float:
        """Describes linear dependency signal from exposition"""
        return self.a * exp + self.b

    def draw_pix(self):
        plt.figure()
        plt.plot(self.expos, self.pix, 'r.', ms=2)
        plt.plot(self.expos, self.avg_pix, '-g')

    def draw_linear(self):
        """Draws linear dependency signal(exposition) and points from approximation"""
        plt.figure()
        plt.errorbar(self.expos, self.avg_pix, yerr=self.d_avg_pix,
                     marker='.', linestyle='', c='red', ms=4, aa=True)
        line_x = np.array([self.expos[0], self.expos[-1]])
        plt.plot(line_x, self.lin_sig(line_x))

    def draw_nonlinearity(self, x_type='exp', each=False, normed=False):
        if x_type == 'exp':
            x = self.expos
        if x_type == 'sig':
            x = self.lin_sig(self.expos)

        plt.figure()
        if normed:
            plt.plot(x, self.nonlin / self.avg_pix, '-g')
        else:
            plt.plot(x, self.nonlin, '-g')
        if each:
            each_pix = self.pix - np.repeat(np.expand_dims(self.lin_sig(self.expos), 1), self.pix.shape[1], axis=1)
            if normed:
                each_pix /= np.repeat(np.expand_dims(self.avg_pix, 1), self.pix.shape[1], axis=1)
            plt.plot(x, each_pix, '-r', lw=1, alpha=0.5)


def process_area_file(filename: str) -> tuple[tuple[int]]:
    rows = []
    columns = []
    with open(filename) as f:
        for line in f:
            match = re.match(r"^([0-9]+)( *\[[0-9-, ]+\])+$", line)
            if match is None:
                raise ValueError(f'Error parsing file {filename}'
                                 f"Line {line} doesn't match format")
            row = int(match.group(1))
            intervals = re.findall(r"\[[0-9-, ]+\]", line)
            cols = set()
            for interval in intervals:
                stripped = interval.strip('[]')
                stripped = stripped.replace(' ', '')
                match = re.match(r"^([0-9]+)-([0-9]+)$", stripped)
                if match is not None:
                    start = int(match.group(1))
                    stop = int(match.group(2))
                    cols.update(tuple(range(start, stop+1)))
                elif re.match(r"^([0-9]+)(,[0-9]+)*$", stripped):
                    cols.update(tuple(map(lambda el: int(el), stripped.split(','))))
                else:
                    raise ValueError('Error decoding pattern'
                                     f'{interval}')
            rows += len(cols) * [row]
            columns += cols

    return (tuple(rows), tuple(columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Nonlinearity tool')
    parser.add_argument('-f', '--folder', required=True,
                        help='Folder, containing FITS-files')
    parser.add_argument('-r', default=False, action='store_true',
                        help='If set, parses all subdirectories')
    parser.add_argument('-m', '--model', required=True,
                        help='File, describing area to analise')
    disp_parser = parser.add_subparsers().add_parser('disp', help='Display options')
    subparsers = disp_parser.add_subparsers(dest='type')
    pix_parser = subparsers.add_parser('pix', help='Dependency for each pixel and average')
    linear_parser = subparsers.add_parser('linear', help='Linear approximation of the dependency')
    nl_parser = subparsers.add_parser('nl', help='Display nonlinearity')
    nl_parser.add_argument('-x', choices=['exp', 'sig'], default='exp',
                  help='X-axis type. Exposition or Signal')
    nl_parser.add_argument('-p', default=False, action='store_true',
                           help='Y-axis (nonlinearity) as part of signal absolute value. ' \
                                'If not set, absolute value is shown')
    nl_parser.add_argument('-a', '--all', default=False, action='store_true',
                           help='Show nonlinarity for each pixel')
    args, rest = parser.parse_known_args()
    arguments = [args]
    while len(rest) > 0 and vars(args) != {'type' : None}:
        args, rest = disp_parser.parse_known_args(rest)
        arguments.append(args)
    if len(rest) > 0:
        disp_parser.parse_args(rest)

    args = arguments[0]
    expos = process_folder(args.folder, args.r)
    indices = process_area_file(args.model)

    area = Area(expos, indices)
    for args in arguments:
        if args.type == 'pix':
            area.draw_pix()
        elif args.type == 'linear':
            area.draw_linear()
        elif args.type == 'nl':
            area.draw_nonlinearity(args.x, args.all, args.p)
    plt.show()
