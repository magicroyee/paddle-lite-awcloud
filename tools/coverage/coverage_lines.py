#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
usage: coverage_lines.py info_file expected
"""
import os
import sys


def get_lines(info_file):
    """

    :param info_file:
    :return:
    """

    hits = .0
    total = .0

    with open(info_file) as info_file:
        for line in info_file:
            line = line.strip()

            if not line.startswith('DA:'):
                continue

            line = line[3:]

            total += 1

            if int(line.split(',')[1]) > 0:
                hits += 1

    if total == 0:
        print 'no data found'
        exit()

    return hits / total


if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit()

    info_file = sys.argv[1]
    expected = float(sys.argv[2])

    if not os.path.isfile(info_file):
        print 'info file {} is not exists, ignored'.format(info_file)
        exit()

    actual = get_lines(info_file)
    actual = round(actual, 3)

    if actual < expected:
        print 'expected >= {} %, actual {} %, failed'.format(
            round(expected * 100, 1), round(actual * 100, 1))

        exit(1)

    print 'expected >= {} %, actual {} %, passed'.format(
        round(expected * 100, 1), round(actual * 100, 1))
