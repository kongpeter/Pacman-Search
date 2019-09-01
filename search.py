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

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    util.raiseNotDefined()

def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE FOR TASK 1 ***"
    startPoint = problem.getStartState()  ## Set the start point

    ## Initialize the deep value
    deepValue = 0

    ## Initialize a new stack
    initialStack = util.Stack()

    while initialStack.isEmpty():  ## From the start point
        deepValue += 1
        initialStack.push(([startPoint], []))  # Data on stack: (visited states, actions)

        while not initialStack.isEmpty():
            (visitedPoint, action) = initialStack.pop()
            currentState = visitedPoint[-1]

            if problem.isGoalState(currentState):
                return action
            elif len(visitedPoint) < deepValue:
                successors = problem.getSuccessors(currentState)
                for (nextPoint, nextAction, totalCost) in successors:
                    if nextPoint not in visitedPoint:
                        visitedPoint.append(nextPoint)   ## append unvisited point
                        action.append(nextAction)
                        initialStack.push((visitedPoint, action))
                        visitedPoint = visitedPoint[:-1]
                        action = action[:-1]

    return action








def waStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has has the weighted (x 2) lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE FOR TASK 2 ***"
    # initialize a new queue
    visitQueue = util.PriorityQueue()

    # create visited list
    pointsList = []

    # push the starting point into queue
    visitQueue.push((problem.getStartState(), [], 0), 0 + 2 * heuristic(problem.getStartState(), problem))

    # pop out the point
    (currentPoint, nextPoint, pathCost) = visitQueue.pop()

    # add the point to visited list
    pointsList.append((currentPoint, pathCost + heuristic(problem.getStartState(), problem)))

    # white goal point is not in list
    while not problem.isGoalState(currentPoint):
        successors = problem.getSuccessors(currentPoint)  # get successor

        for pointInfo in successors:
            visitState = False  # state of visiting
            totalCost = pathCost + pointInfo[2]

            for (visitedState, visitedToCost) in pointsList:
                # if the successor has not been visited, or has a lower cost
                if (pointInfo[0] == visitedState) and (totalCost >= visitedToCost):
                    visitState = True
                    break

            if not visitState:
                # push the point
                visitQueue.push((pointInfo[0], nextPoint + [pointInfo[1]], pathCost + pointInfo[2]),
                                pathCost + pointInfo[2] + heuristic(pointInfo[0], problem))
                # add point to list
                pointsList.append((pointInfo[0], pathCost + pointInfo[2]))

        (currentPoint, nextPoint, pathCost) = visitQueue.pop()

    return nextPoint


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
wastar = waStarSearch
