############################################################
# CMPSC 442: Homework 2
############################################################

student_name = "Dmitri Gordienko"

############################################################
# Imports
############################################################

import collections
import itertools
import math
import random as rndm
import queue as qu
import email
import os
import re
import string
import copy as cp
import sys
import time


############################################################
# Section 1: N-Queens
############################################################


def num_placements_all(n):
    # number of tiles CHOOSE n
    num_tiles = n * n
    return int(math.factorial(num_tiles) / (math.factorial(n) * math.factorial((n * n) - n)))


def num_placements_one_per_row(n):
    return n ** n


def n_queens_valid(board):
    # check that there are no more than one queen on the same row
    if len(set(board)) < len(board):
        return False

    # check that they are not on the same diagonal
    for x in range(len(board)):
        for y in range(x + 1, len(board)):
            if (y - x) == abs(board[x] - board[y]):
                return False

    # if all checks passed return true
    return True


def n_queens_solutions(n):
    for x in range(n):
        for sol_boards in n_queens_helper(n, [x]):
            yield sol_boards
    pass


def n_queens_helper(n, board):
    if n_queens_valid(board):
        if len(board) == n:
            yield board
        else:
            # iterate through all columns that were not added
            for x in [column for column in range(n) if column not in board]:
                if (x != board[-1]) and (x != board[-1] + 1) and (x != board[-1] - 1):
                    solution = list(board)
                    solution.append(x)
                    for result in n_queens_helper(n, solution):
                        if result != ():
                            yield result


############################################################
# Section 2: Lights Out
############################################################


class LightsOutPuzzle(object):

    def __init__(self, board):
        self.board = board
        self.row_len = len(board) - 1
        self.col_len = len(board[0]) - 1

    def get_board(self):
        return list(self.board)

    def perform_move(self, row, col):
        self.board[row][col] = not self.board[row][col]  # toggle
        if row - 1 >= 0:
            self.board[row - 1][col] = not self.board[row - 1][col]
        if col - 1 >= 0:
            self.board[row][col - 1] = not self.board[row][col - 1]
        if row + 1 <= self.row_len:
            self.board[row + 1][col] = not self.board[row + 1][col]
        if col + 1 <= self.col_len:
            self.board[row][col + 1] = not self.board[row][col + 1]

    def scramble(self):
        for row in range(self.row_len + 1):
            for col in range(self.col_len + 1):
                if rndm.random() < 0.5:
                    self.perform_move(row, col)

    def is_solved(self):
        return not any([i for row in self.board for i in row])

    def copy(self):
        return cp.deepcopy(self)

    def successors(self):
        for row in range(self.row_len + 1):
            for col in range(self.col_len + 1):
                successor = self.copy()
                successor.perform_move(row, col)
                yield (row, col), successor

    def find_solution(self):
        q = qu.Queue()
        q.put(self)

        explored = set()
        parent = {self: None}
        moves = {self: None}

        solution = []

        while not q.empty():
            board = q.get()
            explored.add(board.to_tuple())
            if board.is_solved():
                node = board
                while not parent[node] is None:
                    solution.append(tuple(moves[node]))
                    node = parent[node]
                return list(reversed(solution))
            else:
                for move, nextBoard in board.successors():
                    if nextBoard.to_tuple() not in explored:
                        q.put(nextBoard)
                        moves[nextBoard] = move
                        parent[nextBoard] = board
        return None

    def to_tuple(self):
        return tuple(tuple(row) for row in self.get_board())


def create_puzzle(rows, cols):
    return LightsOutPuzzle([[False for row in range(rows)] for col in range(cols)])


############################################################
# Section 3: Linear Disk Movement
############################################################
class LinearDiskMovement(object):

    def __init__(self, length, n, idential=True):
        self.length = length
        self.n = n
        if idential:
            self.cells = [1 if i < n else 0 for i in range(length)]
        else:
            self.cells = [i + 1 if i < n else 0 for i in range(length)]

    def perform_move(self, from_index, to_index):
        self.cells[to_index] = self.cells[from_index]
        self.cells[from_index] = 0

    def copy(self):
        return cp.deepcopy(self)

    def successors(self):
        i = 0
        for i in range(self.length):
            if self.cells[i]:
                if (i + 1 < self.length) and (self.cells[i + 1] == 0):
                    new_disk = self.copy()
                    new_disk.perform_move(i, i + 1)
                    yield tuple((i, i + 1)), new_disk
                if (i - 1 >= 0) and (self.cells[i - 1] == 0):
                    new_disk = self.copy()
                    new_disk.perform_move(i, i - 1)
                    yield tuple((i, i - 1)), new_disk
                if (i + 2 < self.length) and (self.cells[i + 2] == 0) and (self.cells[i + 1] != 0):
                    new_disk = self.copy()
                    new_disk.perform_move(i, i + 2)
                    yield tuple((i, i + 2)), new_disk
                if (i - 2 >= 0) and (self.cells[i - 2] == 0) and (self.cells[i - 1] != 0):
                    new_disk = self.copy()
                    new_disk.perform_move(i, i - 2)
                    yield tuple((i, i - 2)), new_disk

    def is_solved_idential(self):
        return all(self.cells[-1 * self.n:])

    def is_solved_distinct(self):
        return range(self.n, 0, -1) == self.cells[-1 * self.n:]

    def solve_cells(self, identical=True):
        q = qu.Queue()
        q.put(self)
        explored = set()
        parent = {self: None}
        moves = {self: None}
        solutions = []

        while not q.empty():
            board = q.get()
            explored.add(tuple(board.cells))
            if identical and board.is_solved_idential():
                node = board
                while not parent[node] is None:
                    solutions.append(moves[node])
                    node = parent[node]
                return list(reversed(solutions))
            elif not identical and board.is_solved_distinct():
                node = board
                while not parent[node] is None:
                    solutions.append(moves[node])
                    node = parent[node]
                return list(reversed(solutions))
            else:
                for move, next_board in board.successors():
                    if tuple(next_board.cells) not in explored:
                        q.put(next_board)
                        moves[next_board] = move
                        parent[next_board] = board
        return None


def solve_identical_disks(length, n):
    return LinearDiskMovement(length, n).solve_cells()


def solve_distinct_disks(length, n):
    return LinearDiskMovement(length, n, idential=False).solve_cells(identical=False)


############################################################
# Section 4: Feedback
############################################################

feedback_question_1 = """
Approximatly 25 hours
"""

feedback_question_2 = """
Section 3 LinearDiskMovement
"""

feedback_question_3 = """
Section 1 more puzzles like this please
"""
