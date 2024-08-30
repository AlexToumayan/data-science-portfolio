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
from util import Stack
from util import Queue
from util import PriorityQueue
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

def depthFirstSearch(problem: SearchProblem):
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


    stack = Stack()  
    visited = set()  

    start_state = problem.getStartState()  
    stack.push((start_state, []))  

    while not stack.isEmpty():
        state, actions = stack.pop()  

        if problem.isGoalState(state):  
            return actions  

        if state not in visited:  
            visited.add(state)  

            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited:
                    new_actions = actions + [action] 
                    stack.push((successor, new_actions)) 
    return []  

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    queue = Queue()  
    visited = set()  

    start_state = problem.getStartState()  
    queue.push((start_state, []))  

    while not queue.isEmpty():
        state, actions = queue.pop()  

        if problem.isGoalState(state):  
            return actions  

        if state not in visited:  
            visited.add(state)  

            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited:
                    new_actions = actions + [action] 
                    queue.push((successor, new_actions)) 
    return []
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    # priority_queue = PriorityQueue()  
    # visited = set()  

    # start_state = problem.getStartState()  
    # priority_queue.push((start_state, []))  

    # while not priority_queue.isEmpty():
    #     state, actions = priority_queue.pop()  

    #     if problem.isGoalState(state):  
    #         return actions  

    #     if state not in visited:  
    #         visited.add(state)  

    #         for successor, action, step_cost in problem.getSuccessors(state):
    #             if successor not in visited:
    #                 new_actions = actions + [action] 
    #                 priority_queue.push((successor, new_actions)) 
    # return []
# Why did the above generic code work properly for DFS and BFS but not for UCS? 
# What is DFS? : Depth first search is a search algorith where it explores as far as possible along the first branch until it reaches a end state and backtracks. 
# Therefore, we had to store the nodes we already visited so that when it back tracked it could go down a node line it hadnt visited yet until it found the goal state and then returned the goal state once it was reached. However this doesnt return the most "optimal" path. It can, but since it does not take cost into consideration it simply isnt advanced enough for optimization. 
# What is BFS? : Breath first search also down as BROAD search is going to expand all  the nodes on the same level first before going down another node. Meaning, we cannot use a stack because we have to pop the node we just hit if there is another node to expand on the same level. Thus, the queue data structure is more optimal for this job. The queue ds will store a tuple containing a state and the path of actions we took to reach that state. We then pop the front of the queue to expplore the current node we touch. This ensures taht the node we discovered first is expanded first (ie the shallowest node) If that node isnt the goal, then we are going to get the successors. To put it simply, For each successor, if it hasn't been visited before, you append it to the end of the queue. This way, all nodes at the current depth are processed before any nodes at the next depth level are processed. This level-order exploration ensures that BFS finds the shortest path (in terms of the number of actions) to the goal.

# UCS is a bit different, because while BFS may be the most optimal in terms of actions, the number of nodes expanded isnt taking into account the cost of each edge. Thusu, if we have a graph where edges are not = in weight, we need to account for step cost and thus, we need to work on optimizing our path for UCS. 

    priority_queue = PriorityQueue()  
    visited = set()  

    start_state = problem.getStartState()  
    priority_queue.push((start_state, []), 0) # Still return the tuple, but also pass in the priority of the item as the priority queue 
                                              # Data Structure takes in the state the path but also the priority of said path 
    while not priority_queue.isEmpty():
        state, actions = priority_queue.pop()  

        if problem.isGoalState(state):  
            return actions  

        if state not in visited:  
            visited.add(state)  

            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited:
                    new_actions = actions + [action] 
                    #use the getCostOfActions method to return the cost of the path weve just taken. 
                    total_cost = problem.getCostOfActions(new_actions)
                    priority_queue.push((successor, new_actions), total_cost) 
    return []



    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # A* search combines the strengths of Uniform Cost Search (UCS) and Greedy Best-First Search. The key idea is to use a priority queue (like in UCS) to expand nodes in an order that combines the cost to reach the node and a heuristic estimate of the cost to reach the goal from the node.

    # Components of A* Search: 
    # 1. Cost Function (g): The actual cost from the start node to the current node.
    # 2. Heuristic Function (h): An estimate of the cost from the current node to the goal node.
    # 3. Priority Function (f): The sum of the cost function and the heuristic function.f(n) = g(n) + h(n)
    priority_queue = PriorityQueue()  
    visited = {}  

    start_state = problem.getStartState()  
    priority_queue.push((start_state, []), heuristic(start_state, problem)) 

    while not priority_queue.isEmpty():
        state, actions = priority_queue.pop()  

        if problem.isGoalState(state):  
            return actions  

        if state not in visited or problem.getCostOfActions(actions) < visited[state]:  
            visited[state] = problem.getCostOfActions(actions)  

            for successor, action, step_cost in problem.getSuccessors(state):
                new_actions = actions + [action]
                g = problem.getCostOfActions(new_actions)
                h = heuristic(successor, problem)
                f  = g + h 
                priority_queue.push ((successor, new_actions), f)
    return []

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
