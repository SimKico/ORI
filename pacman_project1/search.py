# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of path that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST

    visited = []
    start = problem.getStartState()
    start_node = (start, [])
    nodes = util.Stack()
    nodes.push(start_node)
    the_path = []
    while not nodes.isEmpty():
        current_node, path = nodes.pop()
        print(current_node)
        print(path)
        if current_node not in visited:
            visited.append(current_node)
            the_path = path
            if problem.isGoalState(current_node):
                return path
            else:
                successors = problem.getSuccessors(current_node)

                for succ_state, succ_path, succ_cost in successors:
                    new_path = path + [succ_path]
                    new_node = (succ_state, new_path)
                    nodes.push(new_node)

    return the_path
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST

    visited = []
    start = problem.getStartState()
    start_node = (start, [], 0)
    nodes = util.Queue()
    nodes.push(start_node)
    path = []
    while not nodes.isEmpty():
        current_node, actions, current_cost = nodes.pop()
        print(current_node)
        print(actions)
        if current_node not in visited:
            visited.append(current_node)
            print(visited)
            path = actions
            if problem.isGoalState(current_node):
                return actions
            else:
                succesors = problem.getSuccessors(current_node)

                for succ_state, succ_path, succ_cost in succesors:
                    new_path = actions + [succ_path]
                    new_cost = current_cost + succ_cost
                    new_node = (succ_state, new_path, new_cost)
                    nodes.push(new_node)

    return path
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    nodes = util.PriorityQueue()
    visited = {}
    start = problem.getStartState()
    start_node = (start, [], 0)
    nodes.push(start_node, 0)

    while not nodes.isEmpty():
        current_node, path, current_cost = nodes.pop()

        if (current_node not in visited) or (current_cost < visited[current_node]):
            visited[current_node] = current_cost

            if problem.isGoalState(current_node):
                return path
            else:
                successors = problem.getSuccessors(current_node)

                for succ_node, succ_path, succ_cost in successors:
                    new_path = path + [succ_path]
                    new_cost = current_cost + succ_cost
                    new_node = (succ_node, new_path, new_cost)

                    nodes.update(new_node, new_cost)

    return path
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    nodes = util.PriorityQueue()
    start = problem.getStartState()
    start_node = (start, [], 0)
    visited = []
    the_path = []
    nodes.push(start_node, 0)

    while not nodes.isEmpty():
        current_node, path, current_cost = nodes.pop()
        visited.append((current_node, current_cost))

        if problem.isGoalState(current_node):
            return path
        else:
            succesors = problem.getSuccessors(current_node)

            for succ_state, succ_path, succ_cost in succesors:
                new_path = path + [succ_path]
                new_cost = problem.getCostOfActions(new_path)
                new_node = (succ_state, new_path, new_cost)

                explored = False
                for visit in visited:
                    visited_state, visited_cost = visit

                    if (succ_state == visited_state) and (new_cost == visited_cost):
                        explored = True

                if not explored:
                    nodes.push(new_node, new_cost + heuristic(succ_state, problem))
                    visited.append((succ_state, new_cost))
                    the_path = new_path
    return the_path
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
