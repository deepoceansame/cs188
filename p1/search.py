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

class Node(object):
    def __init__(self, cos, acu_move, acu_cost):
        self.cos = cos
        self.acu_move = acu_move
        self.acu_cost = acu_cost

class Path(object):
    def __init__(self, locations, directions, cost):
        self.locations = locations
        self.directions = directions
        self.cost = cost

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    node = Node([problem.getStartState()], [], 0)
    stack = util.Stack()
    stack.push(node)

    while not stack.isEmpty():
        pop_node = stack.pop()
        cos = pop_node.cos
        acu_move = pop_node.acu_move
        acu_cost = pop_node.acu_cost
        if problem.isGoalState(cos[-1]):
            return acu_move
        for succ in problem.getSuccessors(cos[-1]):
            if succ[0] not in cos:
                newco = succ[0]
                nextCos = cos[:]
                nextCos.append(newco)
                appmove = succ[1]
                appcost = succ[2]
                nextMoves = acu_move[:]
                nextMoves.append(appmove)
                nextCost =  acu_cost+appcost
                stack.push(Node(nextCos, nextMoves, nextCost))
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    node = Node([problem.getStartState()], [], 0)
    que = util.Queue()
    que.push(node)
    visited = [problem.getStartState()]

    while not que.isEmpty():
        pop_node = que.pop()
        cos = pop_node.cos
        visited.append(cos[-1])
        acu_move = pop_node.acu_move
        acu_cost = pop_node.acu_cost
        if problem.isGoalState(cos[-1]):
            return acu_move
        for succ in problem.getSuccessors(cos[-1]):
            nco = succ[0]
            if (nco not in cos) and (nco not in visited):
                visited.append(succ[0])
                newco = succ[0]
                nextCos = cos[:]
                nextCos.append(newco)
                appmove = succ[1]
                appcost = succ[2]
                nextMoves = acu_move[:]
                nextMoves.append(appmove)
                nextCost = acu_cost + appcost
                que.push(Node(nextCos, nextMoves, nextCost))
    return []



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    path = Path([problem.getStartState()], [], 0)

    if problem.isGoalState(problem.getStartState()):
        return path.directions

    queue = util.PriorityQueue()
    queue.push(path, 0)
    visited = [problem.getStartState()]

    while not queue.isEmpty():
        currentPath = queue.pop()
        currentLocation = currentPath.locations[-1]
        if problem.isGoalState(currentLocation):
            return currentPath.directions
        else:
            nextSteps = problem.getSuccessors(currentLocation)
            for nextStep in nextSteps:
                nextLocation = nextStep[0]
                nextDirection = nextStep[1]
                nextCost = nextStep[2]
                if (nextLocation not in currentPath.locations) and (nextLocation not in visited):
                    if not problem.isGoalState(nextLocation):
                        visited.append(nextLocation)
                    nextLocations = currentPath.locations[:]
                    nextLocations.append(nextLocation)
                    nextDirections = currentPath.directions[:]
                    nextDirections.append(nextDirection)
                    nextCosts = currentPath.cost + nextCost
                    nextPath = Path(nextLocations, nextDirections, nextCosts)
                    queue.push(nextPath, nextCosts)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    path = Path([problem.getStartState()], [], 0)

    if problem.isGoalState(problem.getStartState()):
        return path.directions

    queue = util.PriorityQueue()
    queue.push(path, 0)
    visited = set()
    visited.add(problem.getStartState())

    while not queue.isEmpty():
        currentPath = queue.pop()
        currentLocation = currentPath.locations[-1]
        if problem.isGoalState(currentLocation):
            return currentPath.directions
        else:
            nextSteps = problem.getSuccessors(currentLocation)
            for nextStep in nextSteps:
                nextLocation = nextStep[0]
                nextDirection = nextStep[1]
                nextCost = nextStep[2]
                if (nextLocation not in visited):
                    if not problem.isGoalState(nextLocation):
                        visited.add(nextLocation)
                    nextLocations = currentPath.locations[:]
                    nextLocations.append(nextLocation)
                    nextDirections = currentPath.directions[:]
                    nextDirections.append(nextDirection)
                    nextCosts = currentPath.cost + nextCost
                    nextHeuristic = heuristic(nextLocation, problem)
                    nextPath = Path(nextLocations, nextDirections, nextCosts)
                    queue.push(nextPath, nextCosts + nextHeuristic)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
