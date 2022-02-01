############################################################
# CMPSC 442: Homework 3
############################################################

student_name = "Type your full name here."

############################################################
# Imports
############################################################

import random
import copy
import queue
import math


############################################################
# Section 1: Tile Puzzle
############################################################


def create_tile_puzzle(rows, cols):
    boardList = []
    num = 1
    for row in range(rows):
        rowList = []
        for col in range(cols):
            rowList.append(num)
            num += 1
        boardList.append(rowList)

    boardList[-1][-1] = 0
    return TilePuzzle(boardList)


class TilePuzzle(object):

    # Required
    def __init__(self, board):
        self.board = board
        self.row = len(board)
        self.col = len(board[0])
        for x in range(self.row):
            for y in range(self.col):
                if board[x][y] == 0:
                    self.position = (x, y)
        self.sol = self.solved_board()
        self.h = 0
        self.f = 0
        self.g = 0
        self.route = []

    # this function will help the preformance of find_solutions_iddfs by a lot!
    def solved_board(self):
        boardList = []
        num = 1
        for row in range(self.row):
            rowList = []
            for col in range(self.col):
                rowList.append(num)
                num += 1
            boardList.append(rowList)

        boardList[-1][-1] = 0
        return boardList

    def get_board(self):
        return list(self.board)

    def perform_move(self, direction):
        pos = self.position
        if direction.lower() == "up":
            if pos[0] == 0:
                # print("can't move there, try different direction") removed because of scramble()
                return False
            else:
                self.board[pos[0]][pos[1]] = self.board[pos[0] - 1][pos[1]]
                self.board[pos[0] - 1][pos[1]] = 0
                self.position = (pos[0] - 1, pos[1])
                return True
        elif direction.lower() == "down":
            if pos[0] == self.row - 1:
                # print("can't move there, try different direction")
                return False
            else:
                self.board[pos[0]][pos[1]] = self.board[pos[0] + 1][pos[1]]
                self.board[pos[0] + 1][pos[1]] = 0
                self.position = (pos[0] + 1, pos[1])
                return True
        elif direction.lower() == "right":
            if pos[1] == self.col - 1:
                # print("can't move there, try different direction")
                return False
            else:
                self.board[pos[0]][pos[1]] = self.board[pos[0]][pos[1] + 1]
                self.board[pos[0]][pos[1] + 1] = 0
                self.position = (pos[0], pos[1] + 1)
                return True
        elif direction.lower() == "left":
            if pos[1] == 0:
                # print("can't move there, try different direction")
                return False
            else:
                self.board[pos[0]][pos[1]] = self.board[pos[0]][pos[1] - 1]
                self.board[pos[0]][pos[1] - 1] = 0
                self.position = (pos[0], pos[1] - 1)
                return True
        # if input is invalid
        print("\nIllegal move, please type 'up' or 'down' or 'right' or 'left' ")
        return False

    def scramble(self, num_moves):
        moves = ["up", "down", "right", "left"]
        legal_move_count = 0
        while legal_move_count < num_moves:
            if self.perform_move(random.choice(moves)):
                legal_move_count += 1
                # print(self.get_board()) in case I want to see all the legal moves

    def is_solved(self):
        solution = create_tile_puzzle(self.row, self.col)
        if self.board == solution.get_board():
            return True
        else:
            return False

    def copy(self):
        return TilePuzzle(copy.deepcopy(self.board))

    def successors(self):
        for directions in ["up", "down", "left", "right"]:
            p = self.copy()
            if p.perform_move(directions):
                yield directions, p

    # Required
    def find_solutions_iddfs(self):
        is_solution = False
        lim = 0
        while not is_solution:
            for move in self.iddfs_helper(lim, []):
                yield move
                is_solution = True
            lim += 1

    def iddfs_helper(self, limit, moves):
        if self.board == self.sol:
            yield moves
        elif len(moves) < limit:
            for move, puzzle in self.successors():
                for sol in puzzle.iddfs_helper(limit, moves + [move]):
                    yield sol

    # Required
    def find_solution_a_star(self):
        open_set = queue.PriorityQueue()
        open_set.put((self.manhattan_distance(self.board), 0, [], self))  # add the initial state to the queue
        route = set()

        while True:
            node = open_set.get()
            if tuple(tuple(x) for x in node[3].board) in route:
                continue
            else:
                route.add(tuple(tuple(x) for x in node[3].board))
            if node[3].is_solved():
                return node[2]
            for (move, new_p) in node[3].successors():
                if tuple(tuple(x) for x in new_p.board) not in route:
                    open_set.put(
                        (node[1] + 1 + new_p.manhattan_distance(self.board), node[1] + 1, node[2] + [move], new_p))

    def manhattan_distance(self, board1):
        distance = 0

        for i in range(self.row):
            for j in range(self.col):
                if self.board[i][j] != 0:
                    new_row = (self.board[i][j] - 1) / self.col
                    new_col = (self.board[i][j] - 1) % self.col
                    distance += abs(i - new_row) + abs(j - new_col)

        return distance


############################################################
# Section 2: Grid Navigation
############################################################
class GridNavigation(object):

    def __init__(self, start, goal, scene):
        self.pos = start
        self.goal = goal
        self.scene = scene
        self.row = len(scene)
        self.col = len(scene[0])

    def perform_move(self, direction):
        if direction == "up" and self.pos[0] > 0:
            if self.scene[self.pos[0] - 1][self.pos[1]] is False:
                self.pos = (self.pos[0] - 1, self.pos[1])
                return True

        if direction == "down" and self.pos[0] < self.row - 1:
            if self.scene[self.pos[0] + 1][self.pos[1]] is False:
                self.pos = (self.pos[0] + 1, self.pos[1])
                return True

        if direction == "left" and self.pos[1] > 0:
            if self.scene[self.pos[0]][self.pos[1] - 1] is False:
                self.pos = (self.pos[0], self.pos[1] - 1)
                return True

        if direction == "right" and self.pos[1] < self.col - 1:
            if self.scene[self.pos[0]][self.pos[1] + 1] is False:
                self.pos = (self.pos[0], self.pos[1] + 1)
                return True

        if direction == "up-left" and self.pos[0] > 0 and self.pos[1] > 0:
            if self.scene[self.pos[0] - 1][self.pos[1] - 1] is False:
                self.pos = (self.pos[0] - 1, self.pos[1] - 1)
                return True

        if direction == "up-right" and self.pos[0] > 0 and self.pos[1] < self.col - 1:
            if self.scene[self.pos[0] - 1][self.pos[1] + 1] is False:
                self.pos = (self.pos[0] - 1, self.pos[1] + 1)
                return True

        if direction == "down-left" and self.pos[0] < self.row - 1 and self.pos[1] > 0:
            if self.scene[self.pos[0] + 1][self.pos[1] - 1] is False:
                self.pos = (self.pos[0] + 1, self.pos[1] - 1)
                return True

        if direction == "down-right" and self.pos[0] < self.row - 1 and self.pos[1] < self.col - 1:
            if self.scene[self.pos[0] + 1][self.pos[1] + 1] is False:
                self.pos = (self.pos[0] + 1, self.pos[1] + 1)
                return True

    def copy(self):
        return GridNavigation(copy.deepcopy(self.pos), self.goal, self.scene)

    def successors(self):
        for directions in ["up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right"]:
            state = self.copy()
            if state.perform_move(directions):
                yield directions, state.pos, state

    def is_solved(self):
        return self.pos == self.goal

    def euc_dist_heuristic(self):
        return math.sqrt((self.pos[0] - self.goal[0]) ** 2 + (self.pos[1] - self.goal[1]) ** 2)

    def find_path_A_star(self):
        open_set = queue.PriorityQueue()
        open_set.put((self.euc_dist_heuristic(), 0, [self.pos], self))  # add the initial state to queue
        route = set()   # keep route of pos history
        while True:
            if open_set.empty():    # no optimal solution
                return None
            node = open_set.get()   # expand according to priority
            if node[3].pos in route:    # discard the node if having been expanded
                continue
            else:
                route.add(node[3].pos)  # add node.pos to route only if it was expanded
            if node[3].is_solved():  # optimal solution found
                return node[2]
            for (direction, new_pos, new_p) in node[3].successors():
                if new_pos not in route:  # don't add to the queue if the node has been expanded
                    if direction in ["up", "down", "left", "right"]:  # step cost = 1
                        open_set.put((node[1] + 1 + new_p.euc_dist_heuristic(), node[1] + 1, node[2] + [new_pos], new_p))
                    else:  # if cross direction, step cost = sqrt(2)
                        open_set.put((node[1] + math.sqrt(2) + new_p.euc_dist_heuristic(), node[1] + math.sqrt(2),
                                      node[2] + [new_pos], new_p))


def find_path(start, goal, scene):
    g = GridNavigation(start, goal, scene)
    return g.find_path_A_star()


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################

class LinearDiskMovement(object):

    def __init__(self, n, length, disks):
        self.n = n
        self.length = length
        self.disks = list(disks)
        self.g = 0
        self.h = 0
        self.f = 0
        self.route = []

    def successors(self):
        for x in range(len(self.disks)):
            if self.disks[x]:
                if x + 1 < self.length:
                    if self.disks[x + 1] == 0:
                        replace = list(self.disks)
                        disk = replace[x]
                        replace[x] = 0
                        replace[x + 1] = disk
                        yield (x, x + 1), LinearDiskMovement(self.n, self.length, replace)

                if x + 2 < self.length:
                    if self.disks[x + 2] == 0 and self.disks[x + 1] != 0:
                        replace = list(self.disks)
                        disk = replace[x]
                        replace[x] = 0
                        replace[x + 2] = disk
                        yield (x, x + 2), LinearDiskMovement(self.n, self.length, replace)

                if x - 1 >= 0:
                    if self.disks[x - 1] == 0:
                        replace = list(self.disks)
                        disk = replace[x]
                        replace[x] = 0
                        replace[x - 1] = disk
                        yield (x, x - 1), LinearDiskMovement(self.n, self.length, replace)

                if x - 2 >= 0:
                    if self.disks[x - 2] == 0 and self.disks[x - 1] != 0:
                        replace = list(self.disks)
                        disk = replace[x]
                        replace[x] = 0
                        replace[x - 2] = disk
                        yield (x, x - 2), LinearDiskMovement(self.n, self.length, replace)

    def heuristic(self, goal):
        pos = {}
        for i, j in enumerate(goal):
            pos[j] = i

        distance = 0
        for x, y in enumerate(self.disks):
            distance += abs(x - pos[y])

        return distance


def solve_distinct_disks(length, n):
    open_set = set()
    start = [x + 1 for x in range(n)]
    for x in range(length - n):
        start.append(0)
    goal = list(reversed(copy.deepcopy(start)))

    if start == goal:
        return [()]

    a = LinearDiskMovement(n, length, start)
    open_set.add(a)

    closed_set = set()
    a.h = a.heuristic(goal)

    while open_set:
        current = min(open_set, key=lambda x: x.f)
        if current.disks == goal:
            return current.route
        open_set.remove(current)

        for move, disk in current.successors():
            if disk.disks == goal:
                disk.route = current.route + [move]
                return disk.route

            disk.g = current.g + current.heuristic(disk.disks)
            disk.h = disk.heuristic(goal)
            disk.f = disk.g + disk.h

            go = True
            for loc in open_set:
                if loc.disks == disk.disks and loc.f < disk.f:
                    go = False
                    continue
            for loc in closed_set:
                if loc.disks == disk.disks and loc.f < disk.f:
                    go = False
                    continue

            if go:
                open_set.add(disk)
                disk.route = current.route + [move]

        closed_set.add(current)


############################################################
# Section 4: Dominoes Game
############################################################

def create_dominoes_game(rows, cols):
    return DominoesGame([[False for row in range(rows)] for col in range(cols)])


class DominoesGame(object):

    # Required
    def __init__(self, board):
        self.board = board
        self.row = len(board)
        self.col = len(board[0])

    def get_board(self):
        return self.board

    def reset(self):
        self.board = [[False for row in range(self.row)] for col in range(self.col)]

    def is_legal_move(self, row, col, vertical):
        if vertical:
            if row == self.row - 1:
                return False
            if self.board[row + 1][col] or self.board[row][col]:
                return False
        else:
            if col == self.col - 1:
                return False
            if self.board[row][col + 1] or self.board[row][col]:
                return False
        return True

    def legal_moves(self, vertical):
        for row in range(self.row):
            for col in range(self.col):
                if self.is_legal_move(row, col, vertical):
                    yield row, col

    def perform_move(self, row, col, vertical):
        if self.is_legal_move(row, col, vertical):
            if vertical:
                self.board[row + 1][col] = True
            else:
                self.board[row][col + 1] = True
            self.board[row][col] = True

    def game_over(self, vertical):
        if list(self.legal_moves(vertical)):
            return False
        return True

    def copy(self):
        return DominoesGame(copy.deepcopy(self.board))

    def successors(self, vertical):
        for x, y in list(self.legal_moves(vertical)):
            cpBoard = self.copy()
            cpBoard.perform_move(x, y, vertical)
            yield (x, y), cpBoard

    def get_random_move(self, vertical):
        return random.choice(tuple(self.legal_moves(vertical)))

    # Required
    def get_best_move(self, vertical, limit):
        inf = float('inf')
        neg_inf = -inf
        return self.max_val(neg_inf, inf, None, vertical, limit)

    def min_val(self, alpha, beta, m, vertical, limit):
        ver = list(self.successors(vertical))
        hor = list(self.successors(not vertical))

        if limit == 0 or self.game_over(vertical):
            return m, len(hor) - len(ver), 1

        path_cost = float('inf')
        num_visited = 0
        current = m
        for pos, child in ver:
            move, temp, count = child.max_val(alpha, beta, pos, not vertical, limit - 1)
            num_visited += count
            if temp < path_cost:
                path_cost = temp
                current = pos
            if path_cost <= alpha:
                return current, path_cost, num_visited
            beta = min(beta, path_cost)

        return current, path_cost, num_visited

    def max_val(self, alpha, beta, m, vertical, limit):
        ver = list(self.successors(vertical))
        hor = list(self.successors(not vertical))

        if limit == 0 or self.game_over(vertical):
            return m, len(ver) - len(hor), 1

        path_cost = -float('inf')
        num_visited = 0
        current = m
        for pos, child in ver:
            move, temp, count = child.min_val(alpha, beta, pos, not vertical, limit - 1)
            num_visited += count
            if temp > path_cost:
                path_cost = temp
                current = pos
            if path_cost >= beta:
                return current, path_cost, num_visited
            alpha = min(alpha, path_cost)

        return current, path_cost, num_visited


############################################################
# Section 5: Feedback
############################################################

feedback_question_1 = """
Way to long to count... 40+
"""

feedback_question_2 = """
A star and dominoes game
"""

feedback_question_3 = """
less problems, more time. This can not go on like that, those are insanely hard to complete
in the given time considering we have other 400-level classes. PLEASE!!!
"""
