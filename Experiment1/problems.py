import copy

from node import Node


class FifteensNode(Node):
    """Extends the Node class to solve the 15 puzzle.

    Parameters
    ----------
    parent : Node, optional
        The parent node. It is optional only if the input_str is provided. Default is None.

    g : int or float, optional
        The cost to reach this node from the start node : g(n).
        In this puzzle it is the number of moves to reach this node from the initial configuration.
        It is optional only if the input_str is provided. Default is 0.

    board : list of lists
        The two-dimensional list that describes the state. It is a 4x4 array of values 0, ..., 15.
        It is optional only if the input_str is provided. Default is None.

    input_str : str
        The input string to be parsed to create the board.
        The argument 'board' will be ignored, if input_str is provided.
        Example: input_str = '1 2 3 4\n5 6 7 8\n9 10 0 11\n13 14 15 12' # 0 represents the empty cell

    Examples
    ----------
    Initialization with an input string (Only the first/root construction call should be formatted like this):
    >>> n = FifteensNode(input_str=initial_state_str)
    >>> print(n)
      5  1  4  8
      7     2 11
      9  3 14 10
      6 13 15 12

    Generating a child node (All the child construction calls should be formatted like this) ::
    >>> n = FifteensNode(parent=p, g=p.g+c, board=updated_board)
    >>> print(n)
      5  1  4  8
      7  2    11
      9  3 14 10
      6 13 15 12

    """

    def __init__(self, parent=None, g=0, board=None, input_str=None):
        # NOTE: You shouldn't modify the constructor
        if input_str:
            self.board = []
            for i, line in enumerate(filter(None, input_str.splitlines())):
                self.board.append([int(n) for n in line.split()])
        else:
            self.board = board

        super(FifteensNode, self).__init__(parent, g)

    def generate_children(self):
        """Generates children by trying all 4 possible moves of the empty cell.
           通过尝试空单元格的所有4种可能的移动来生成子单元。

        Returns
        -------
            children : list of Nodes
                The list of child nodes.
        """

        # You should use self.board to produce children. Don't forget to create a new board for each child
        # e.g you can use copy.deepcopy function from the standard library.
        children = []
        # Get the empty cell's location
        x, y = 0, 0
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):  # provided board is n*n
                if self.board[i][j] == 0:
                    x, y = i, j

        # Generate four
        if x != len(self.board[0]) - 1:  # The empty cell & cell on its right
            child_board = copy.deepcopy(self.board)
            child_board[x][y], child_board[x + 1][y] = child_board[x + 1][y], child_board[x][y]
            children.append(FifteensNode(g=self.g, parent=self, board=child_board))
        if x != 0:  # The empty cell & cell on its left
            child_board = copy.deepcopy(self.board)
            child_board[x][y], child_board[x - 1][y] = child_board[x - 1][y], child_board[x][y]
            children.append(FifteensNode(g=self.g, parent=self, board=child_board))
        if y != len(self.board) - 1:  # The empty cell & cell under it
            child_board = copy.deepcopy(self.board)
            child_board[x][y], child_board[x][y + 1] = child_board[x][y + 1], child_board[x][y]
            children.append(FifteensNode(g=self.g, parent=self, board=child_board))
        if y != 0:  # The empty cell & cell on it
            child_board = copy.deepcopy(self.board)
            child_board[x][y], child_board[x][y - 1] = child_board[x][y - 1], child_board[x][y]
            children.append(FifteensNode(g=self.g, parent=self, board=child_board))

        return children

    def is_goal(self):
        """Decides whether this search state is the final state of the puzzle.

        Returns
        -------
            is_goal : bool
                True if this search state is the goal state, False otherwise.
        """

        # You should use self.board to decide.
        goaled = True
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):  # provided board is n*n
                # Ignore the empty cell on the last position.
                if i == len(self.board) - 1 and j == len(self.board[i]) - 1 and self.board[i][j] == 0:
                    break
                if self.board[i][j] != len(self.board[i]) * i + j + 1:
                    # print(self.board[i][j], end=" ")
                    goaled = False
                    break
        return goaled

    def evaluate_heuristic(self):
        """Heuristic function h(n) that estimates the minimum number of moves
        required to reach the goal state from this node.

        Returns
        -------
            h : int or float
                The heuristic value for this state.
        """

        # You may want to use self.board here.
        heuristic = 0
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):  # provided board is n*n
                if self.board[i][j] == 0:
                    continue
                if self.board[i][j] != len(self.board[i]) * i + j + 1:
                    # heuristic += 1
                    # If board[i][j] == num, and num should be placed in board[m][n],
                    # then heuristic += abs(m-i)+abs(n-j)
                    num = self.board[i][j]
                    target_x = num / len(self.board)
                    target_y = num % len(self.board)
                    heuristic += abs(target_x - i) + abs(target_y - j)
        return heuristic

    def _get_state(self):
        """Returns an hashable representation of this search state.

        Returns
        -------
            state: tuple
                The hashable representation of the search state
        """
        # NOTE: You shouldn't modify this method.
        return tuple([n for row in self.board for n in row])

    def __str__(self):
        """Returns the string representation of this node.

        Returns
        -------
            state_str : str
                The string representation of the node.
        """
        # NOTE: You shouldn't modify this method.
        sb = []  # String builder
        for row in self.board:
            for i in row:
                sb.append(' ')
                if i == 0:
                    sb.append('  ')
                else:
                    if i < 10:
                        sb.append(' ')
                    sb.append(str(i))
            sb.append('\n')
        return ''.join(sb)


class SuperqueensNode(Node):
    """Extends the Node class to solve the Superqueens problem.

    Parameters
    ----------
    parent : Node, optional
        The parent node. Default is None.

    g : int or float, optional
        The cost to reach this node from the start node : g(n).
        In this problem it is the number of pairs of superqueens that can attack each other in this state configuration.
        Default is 1.

    queen_positions : list of pairs
        The list that stores the x and y positions of the queens in this state configuration.
        Example: [(q1_y,q1_x),(q2_y,q2_x)]. Note that the upper left corner is the origin and y increases downward
        Default is the empty list [].
        ------> x
        |
        |
        v
        y

    n : int
        The size of the board (n x n)

    Examples
    ----------
    Initialization with a board size (Only the first/root construction call should be formatted like this):
    >>> n = SuperqueensNode(n=4)
    >>> print(n)
         .  .  .  .
         .  .  .  .
         .  .  .  .
         .  .  .  .

    Generating a child node (All the child construction calls should be formatted like this):
    >>> n = SuperqueensNode(parent=p, g=p.g+c, queen_positions=updated_queen_positions, n=p.n)
    >>> print(n)
         Q  .  .  .
         .  .  .  .
         .  .  .  .
         .  .  .  .

    """

    def __init__(self, parent=None, g=0, queen_positions=[], n=1):
        # NOTE: You shouldn't modify the constructor
        self.queen_positions = queen_positions
        self.n = n
        super(SuperqueensNode, self).__init__(parent, g)

    def generate_children(self):
        """Generates children by adding a new queen.

        Returns
        -------
            children : list of Nodes
                The list of child nodes.
        """
        # You should use self.queen_positions and self.n to produce children.
        # Don't forget to create a new queen_positions list for each child.
        # You can use copy.deepcopy function from the standard library.
        children = []
        y_occupied = []
        x_now = len(self.queen_positions)
        for position in self.queen_positions:
            had_y, _ = position
            y_occupied.append(had_y)

        for i in range(self.n):
            # hard condition: not in same rows or same columns
            if i not in y_occupied:
                new_queen_positions = copy.deepcopy(self.queen_positions)
                new_position = (i, x_now)
                new_queen_positions.append(new_position)
                children.append(SuperqueensNode(parent=self, g=self.g, n=self.n,
                                                queen_positions=new_queen_positions))

        return children

    def is_goal(self):
        """Decides whether all the queens are placed on the board.

        Returns
        -------
            is_goal : bool
                True if all the queens are placed on the board, False otherwise.
        """

        # You should use self.queen_positions and self.n to decide.
        # def meet_the_hard_condition(position1, position2):
        #     # x, y may equals zero, add 1
        #     y, x = position1
        #     another_y, another_x = position2
        #     print(y - another_y, " ", x - another_x)
        #     return not (y == another_y or x == another_x or y - another_y == x - another_x)

        # print(len(self.queen_positions), self.n)
        if len(self.queen_positions) != self.n:
            return False

        goaled = True
        for i, position in enumerate(self.queen_positions):
            y, x = position
            if y >= self.n or x >= self.n:
                goaled = False

        return goaled

    def evaluate_heuristic(self):
        """Heuristic function h(n) that estimates the minimum number of conflicts required to reach the final state.

        Returns
        -------
            h : int or float
                The heuristic value for this state.
        """
        # If you want to design a heuristic for this problem, you should use self.queen_positions and self.n.
        if len(self.queen_positions) == 0:
            return self.g
        child_position = self.queen_positions[len(self.queen_positions) - 1]
        new_g = self.g
        y, x = child_position
        for position_i in self.queen_positions:
            y_i, x_i = position_i
            if y == y_i and x == x_i:
                continue
            # L/\
            dy, dx = abs(y - y_i), abs(x - x_i)
            new_g += 1 if (dy == dx) else 0
            new_g += 1 if (dx == 2 and dy == 1) or (dx == 1 and dy == 2) else 0
        return new_g

    def _get_state(self):
        """Returns an hashable representation of this search state.

        Returns
        -------
            state: tuple
                The hashable representation of the search state
        """
        # NOTE: You shouldn't modify this method.
        return tuple(self.queen_positions)

    def __str__(self):
        """Returns the string representation of this node.

        Returns
        -------
            state_str : str
                The string representation of the node.
        """
        # NOTE: You shouldn't modify this method.
        sb = [[' . '] * self.n for i in range(self.n)]  # String builder
        for i, j in self.queen_positions:
            sb[i][j] = ' Q '
        return '\n'.join([''.join(row) for row in sb])
