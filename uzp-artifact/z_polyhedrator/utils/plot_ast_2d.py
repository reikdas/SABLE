#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script plots an AST file into a 2D plane. AST files are expected to be in
# the following format
#
# Row\tCol\tN\tI\tJ
# 0\t0\t2\t1\t0
# 0\t0\t2\t0\t1
# 0\t1\t2\t1\t0
# 0\t1\t2\t0\t1
# 1\t0\t2\t1\t0
# 1\t0\t2\t0\t1
#
# where each line represents an AST.
#
# The first line is the header and the following lines are the ASTs. Each AST is
# defined by the following parameters:
#
# - Row: The row of the first point of the AST
# - Col: The column of the first point of the AST
# - N: The number of points of the AST
# - I: The row increment between points of the AST
# - J: The column increment between points of the AST
#
# An AST file is generated using the following syntax:
#
# z_polyhedrator patterns.txt matrix.mtx --print-ast-list > ast_file.txt
#
# where `z_polyhedrator` is the path to the z_polyhedrator executable, `patterns.txt` is
# the path to the patterns file, `matrix.mtx` is the path to the matrix file and
# `ast_file.txt` is the path to the AST file to be generated.
#
# The script is then executed using the following syntax:
#
# python3 plot_ast_2d.py ast_file.txt [-o output.pdf]
#
# where `ast_file.txt` is the path to the AST file to be plotted and
# `output.pdf` is the path to the output PDF file.
#
# If the output file already exists, it will be overwritten.
#
# If the output file is not specified, the AST file will be saved in your
# current working directory with the same name as the AST file but with the
# `.pdf` extension.
#
# The script will create a temporary directory in your system's temporary
# directory. This directory will be deleted when the script finishes.

__version__ = 'v0.0.1'

__email__ = 'i.amatria@udc.es'
__author__ = 'Iñaki Amatria-Barral'

__license__ = 'I addere to any license you want to use this code under'

import os
import shutil
import PyPDF2
import argparse
import tempfile
import matplotlib
import charset_normalizer

matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor

class AST:
    def __init__(self, row, col, n, i, j):
        self._row = row
        self._col = col
        self._n = n
        self._i = i
        self._j = j

    def max_row(self):
        return self._row + (self._n - 1) * self._i

    def max_col(self):
        return self._col + (self._n - 1) * self._j

    def is_in_block(self, min_x, max_x, min_y, max_y):
        for k in range(self._n):
            x = self._col + k * self._j
            y = self._row + k * self._i
            if x >= min_x and x < max_x and y >= min_y and y < max_y:
                return True
        for k in range(self._n - 1):
            start_x = self._col + k * self._j
            start_y = self._row + k * self._i
            end_x = self._col + (k + 1) * self._j
            end_y = self._row + (k + 1) * self._i

            if start_x > end_x:
                start_x, end_x = end_x, start_x
            if start_y > end_y:
                start_y, end_y = end_y, start_y

            for x in range(start_x, end_x + 1):
                for y in range(start_y, end_y + 1):
                    if x >= min_x and x < max_x and y >= min_y and y < max_y:
                        return True
        return False

    def plot(self, color, ax):
        for k in range(self._n):
            surface = [
                (self._col + k * self._j, self._row + k * self._i),
                (self._col + k * self._j + 1, self._row + k * self._i),
                (self._col + k * self._j + 1, self._row + k * self._i + 1),
                (self._col + k * self._j, self._row + k * self._i + 1)
            ]
            ax.add_patch(
                plt.Polygon(surface, facecolor=color, alpha=0.5, zorder=0)
            )

        ax.scatter(self._col + 0.5, self._row + 0.5, color=color, zorder=2)
        for k in range(1, self._n):
            ax.scatter(
                self._col + k * self._j + 0.5,
                self._row + k * self._i + 0.5,
                color=color,
                zorder=2
            )

        for k in range(self._n - 1):
            ax.plot(
                (
                    self._col + k * self._j + 0.5,
                    self._col + (k + 1) * self._j + 0.5
                ),
                (
                    self._row + k * self._i + 0.5,
                    self._row + (k + 1) * self._i + 0.5
                ),
                color=color,
                zorder=1
            )

class ASTReader:
    AST_MAGIC_HEADER = 'RowColNIJ'

    def __init__(self, ast_file):
        self._ast_file = ast_file

    def read(self):
        if not os.path.isfile(self._ast_file):
            raise FileNotFoundError(f'AST file {self._ast_file} does not exist')

        results = str(charset_normalizer.from_path(self._ast_file).best())
        raw_asts = [line.strip() for line in results.split('\n')]
        raw_asts = [line for line in raw_asts if line != '']

        if len(raw_asts) == 0:
            raise ValueError(f'AST file {self._ast_file} is empty')

        if ''.join(raw_asts[0].split()) != ASTReader.AST_MAGIC_HEADER:
            raise ValueError(
                f'AST file {self._ast_file} does not have a valid header'
            )
        raw_asts = raw_asts[1:]

        asts = []
        for raw_ast in raw_asts:
            tmp_raw_ast = raw_ast.split()

            if len(tmp_raw_ast) != 5:
                raise ValueError(f'`{raw_ast}` is not a valid AST')
            if any(not self._is_int(value) for value in tmp_raw_ast):
                raise ValueError(f'`{raw_ast}` is not a valid AST')

            row, col, n, i, j = [int(value) for value in tmp_raw_ast]
            asts.append(AST(row, col, n, i, j))

        return asts

    def _is_int(self, value):
        try:
            int(value)
            return True
        except ValueError:
            return False

class ASTPlotter:
    STRIDE_X = 25
    STRIDE_Y = 25

    def __init__(self, asts, output_file):
        self._asts = asts
        self._output_file = output_file

        self._max_x_idx = max(ast.max_col() for ast in self._asts)
        self._max_y_idx = max(ast.max_row() for ast in self._asts)

        self._max_x = ASTPlotter.STRIDE_X * int(
            (self._max_x_idx + ASTPlotter.STRIDE_X) / ASTPlotter.STRIDE_X
        )
        self._max_y = ASTPlotter.STRIDE_Y * int(
            (self._max_y_idx + ASTPlotter.STRIDE_Y) / ASTPlotter.STRIDE_Y
        )

    def plot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._plot_blocks(tmp_dir)
            self._merge_blocks_into_rows(tmp_dir)
            self._merge_rows_into_final_image(tmp_dir)

            if os.path.dirname(self._output_file) != '':
                os.makedirs(os.path.dirname(self._output_file), exist_ok=True)
            shutil.move(os.path.join(tmp_dir, 'ast.pdf'), self._output_file)

    def _plot_blocks(self, tmp_dir):
        total_x = int(self._max_x / ASTPlotter.STRIDE_X)
        total_y = int(self._max_y / ASTPlotter.STRIDE_Y)

        description = 'Plotting PDFs'
        progress_bar = tqdm(total=total_x * total_y, desc=description)
        with ProcessPoolExecutor() as ex:
            for x in range(0, self._max_x, ASTPlotter.STRIDE_X):
                for y in range(0, self._max_y, ASTPlotter.STRIDE_Y):
                    save_path = os.path.join(tmp_dir, f'block_{x}_{y}.pdf')
                    ex.submit(
                        self._plot_block,
                        x,
                        y,
                        save_path
                    ).add_done_callback(lambda _: progress_bar.update())
        progress_bar.close()

    def _merge_blocks_into_rows(self, tmp_dir):
        total_merges = int(self._max_y / ASTPlotter.STRIDE_Y)

        description = 'Merging PDFs'
        progress_bar = tqdm(total=total_merges, desc=description)
        with ProcessPoolExecutor() as ex:
            for y in range(0, self._max_y, ASTPlotter.STRIDE_Y):
                save_path = os.path.join(tmp_dir, f'row_{y}.pdf')
                ex.submit(
                    self._merge_blocks_into_row,
                    y,
                    tmp_dir,
                    save_path
                ).add_done_callback(lambda _: progress_bar.update())
        progress_bar.close()

    def _merge_rows_into_final_image(self, tmp_dir):
        total_reductions = int(np.log2(self._max_y / ASTPlotter.STRIDE_Y)) + 1

        description = 'Building final PDF'
        progress_bar = tqdm(total=total_reductions, desc=description)
        with ProcessPoolExecutor() as ex:
            for k in range(total_reductions):
                stride = ASTPlotter.STRIDE_Y * (2 ** (k + 1))
                futures = [
                    ex.submit(self._merge_rows_into_row, k, j, tmp_dir)
                    for j in range(0, self._max_y, stride)
                ]
                _ = [future.result() for future in futures]
                progress_bar.update()
        progress_bar.close()

    def _plot_block(self, x, y, save_path):
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(self._asts)))
        ast_color= {
            (ast._n, ast._i, ast._j): color
            for ast, color in zip(self._asts, colors)
        }

        max_x = x + ASTPlotter.STRIDE_X
        max_y = y + ASTPlotter.STRIDE_Y
        asts = [
            ast for ast in self._asts
            if ast.is_in_block(x, max_x, y, max_y)
        ]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for ast in asts:
            ast.plot(ast_color[(ast._n, ast._i, ast._j)], ax)

        if len(asts) == 0 and not (x == 0 and y == 0):
            plt.close()
            return

        plt.gca().set_axis_off()
        plt.subplots_adjust(
            top=1,
            bottom=0,
            right=1,
            left=0,
            hspace=0,
            wspace=0
        )
        plt.margins(0, 0)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(x, max_x)
        ax.set_ylim(y, max_y)

        ax.invert_yaxis()

        ax.set_aspect('equal')

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _merge_blocks_into_row(self, y, tmp_dir, save_path):
        reference_block = os.path.join(tmp_dir, f'block_0_0.pdf')
        width, height = self._get_block_dimensions(reference_block)

        row = PyPDF2.PageObject.create_blank_page(
            width=width * (self._max_x_idx + 1) / ASTPlotter.STRIDE_X,
            height=height
        )

        for x in range(0, self._max_x, ASTPlotter.STRIDE_X):
            block = os.path.join(tmp_dir, f'block_{x}_{y}.pdf')
            if not os.path.isfile(block):
                continue

            pdf = PyPDF2.PdfReader(block)
            page = pdf.pages[0]

            page.add_transformation(
                PyPDF2.Transformation().translate(
                    int(x / ASTPlotter.STRIDE_X) * width, 0
                )
            )
            page.mediabox = row.mediabox

            row.merge_page(page)

        pdf = PyPDF2.PdfWriter()
        pdf.add_page(row)
        pdf.write(save_path)

    def _merge_rows_into_row(self, k, j, tmp_dir):
        reference_block = os.path.join(tmp_dir, f'block_0_0.pdf')
        width, height = self._get_block_dimensions(reference_block)

        total_reductions = int(np.log2(self._max_y / ASTPlotter.STRIDE_Y)) + 1

        top_row_idx, bottom_row_idx = j, j + (2 ** k) * ASTPlotter.STRIDE_Y

        top_file = os.path.join(tmp_dir, f'row_{k - 1}_')
        bottom_file = os.path.join(tmp_dir, f'row_{k - 1}_')
        if k == 0:
            top_file = os.path.join(tmp_dir, f'row_')
            bottom_file = os.path.join(tmp_dir, f'row_')
        top_file = f'{top_file}{top_row_idx}.pdf'
        bottom_file = f'{bottom_file}{bottom_row_idx}.pdf'

        row = PyPDF2.PageObject.create_blank_page(
            width=width * (self._max_x_idx + 1) / ASTPlotter.STRIDE_X,
            height=height * 2 * (2 ** k)
        )

        if os.path.isfile(top_file):
            pdf = PyPDF2.PdfReader(top_file)
            page = pdf.pages[0]

            page.add_transformation(
                PyPDF2.Transformation().translate(0, height * (2 ** k))
            )
            page.mediabox = row.mediabox

            row.merge_page(page)
        if os.path.isfile(bottom_file):
            pdf = PyPDF2.PdfReader(bottom_file)
            page = pdf.pages[0]

            page.add_transformation(
                PyPDF2.Transformation().translate(0, 0)
            )
            page.mediabox = row.mediabox

            row.merge_page(page)

        save_path = os.path.join(tmp_dir, f'row_{k}_{j}.pdf')
        if k == total_reductions - 1:
            save_path = os.path.join(tmp_dir, f'ast.pdf')

            row.mediabox.lower_left = (
                row.mediabox.lower_left[0],
                row.mediabox.upper_left[1]
                    - height * (self._max_y_idx + 1) / ASTPlotter.STRIDE_Y
            )
            row.mediabox.lower_right = (
                row.mediabox.lower_right[0],
                row.mediabox.upper_right[1]
                    - height * (self._max_y_idx + 1) / ASTPlotter.STRIDE_Y
            )

        pdf = PyPDF2.PdfWriter()
        pdf.add_page(row)
        pdf.write(save_path)

    def _get_block_dimensions(self, block):
        pdf = PyPDF2.PdfReader(block)
        page = pdf.pages[0]
        return page.mediabox.width, page.mediabox.height

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='AST Plotter',
        description='A Python utility to plot ASTs in a 2D plane'
    )

    parser.add_argument(
        'input_ast_file',
        type=str,
        help='the AST file to plot'
    )
    parser.add_argument(
        '-o',
        '--output-pdf',
        type=str,
        required=False,
        metavar='output_pdf',
        help='the output PDF file'
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    args = parser.parse_args()

    ast_file = args.input_ast_file
    output_file = os.path.basename(ast_file)
    output_file = f'{os.path.splitext(output_file)[0]}.pdf'
    if args.output_pdf:
        output_file = args.output_pdf

    asts = ASTReader(ast_file).read()
    ASTPlotter(asts, output_file).plot()
