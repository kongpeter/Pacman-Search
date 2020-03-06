# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DefensiveAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########
class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)

        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class DummyAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
        CaptureAgent.registerInitialState(self, gameState)

        # Go home
        self.start = gameState.getAgentPosition(self.index)
        self.catchState = False

        currentFood = self.getFood(gameState).asList()

        value = 0
        for food in currentFood:
            value += food[1]

        average = value / len(currentFood)
        self.average = average

        '''
    Your initialization code goes here, if you need any.
    '''

    ################
    # Main Methods #
    ################

    def chooseAction(self, gameState):
        """
    Picks among actions with highest Q(s,a) same one as in baseline Team.
    """
        # If the value returns smaller eval deems better
        foodLeft = len(self.getFood(gameState).asList())

        actions = gameState.getLegalActions(self.index)
        actionList = []
        qValuesList = []
        myState = gameState.getAgentState(self.index)

        if (myState.numCarrying >= 4):
            goalPoint = self.start
            bestAction = self.aStarSearch(gameState, goalPoint)
            # self.checkCatch(gameState.generateSuccessor(self.index, bestAction))
        else:
            for action in actions:
                if action is not 'Stop':
                    actionList.append(action)
                    currentState = self.getCurrentObservation()
                    successorState = gameState.generateSuccessor(self.index, action)
                    qValue = self.evaluate(currentState, successorState, action)
                    qValuesList.append(qValue)
                    maxValue = max(qValuesList)
                    maxIndex = qValuesList.index(maxValue)
                    bestAction = actionList[maxIndex]

        return bestAction

    def getDistance(self, gameState, successorPos, goalPoint):
        minDistance = 9999
        if goalPoint is None:
            foodList = self.getFood(gameState).asList()
            capsuleList = self.getCapsules(gameState)
            eatingList = foodList + capsuleList

            if successorPos in eatingList:
                # distance reward is -100
                distance = 0 + (-100)
            else:  # find minimum distance to food
                # distance = min([self.getMazeDistance(successorPos, food) for food in foodList])
                for food in eatingList:
                    if self.getMazeDistance(successorPos, food) < minDistance:
                        minDistance = self.getMazeDistance(successorPos, food)
                distance = minDistance

        else:
            # distance to start point
            distance = self.getMazeDistance(successorPos, self.start)

            if self.getMazeDistance(gameState.getAgentPosition(self.index), self.start) > 1:
                if successorPos == self.start:
                    distance = 9999999  # being catch and go home, so this distance should be the biggest one
        return distance

    def checkCatch(self, successor):
        position = successor.getAgentPosition(self.index)
        myState = successor.getAgentState(self.index)

        minDistance = 9999
        opponentGhost = []

        opponentsIndices = self.getOpponents(successor)
        for opponentIndex in opponentsIndices:
            opponent = successor.getAgentState(opponentIndex)

            if not opponent.isPacman and opponent.getPosition() is not None:
                oppentPos = opponent.getPosition()
                disToOppent = self.getMazeDistance(position, oppentPos)
                if disToOppent < minDistance:
                    minDistance = disToOppent
                    opponentGhost.append(opponent)

        if len(opponentGhost) > 0 and minDistance <= 2:
            # ghost scared:
            if opponentGhost[-1].scaredTimer > 0:
                self.catchState = False
            else:
                self.catchState = True
        else:
            self.catchState = False

    def aStarSearch(self, gameState, goalPoint):

        ############### initialization ##################
        explored = []
        exploring = util.PriorityQueue()
        legalAction = []
        done = False
        exploring.push([gameState, []], 0)
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        eatingList = foodList + capsuleList
        #################################################
        while not done:
            popItem = exploring.pop()
            currentState = popItem[0]
            beforeAction = popItem[1]
            currentPos = currentState.getAgentPosition(self.index)

            ## when find the needed point ##
            if currentPos == goalPoint or (currentPos in eatingList and not self.catchState):
                done = True
                return beforeAction[0]

            ## avoid duplicate exploration ##
            if currentPos in explored:
                continue
            else:
                explored.append(currentPos)
                legalAction = currentState.getLegalActions(self.index)

            for action in legalAction:
                successor = currentState.generateSuccessor(self.index, action)
                successorPos = successor.getAgentPosition(self.index)
                # ghost Pos
                enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
                ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
                if len(ghosts) > 0:
                    dists = [self.getMazeDistance(successorPos, a.getPosition()) for a in ghosts]

                    if min(dists) < 5:
                        fx = self.getDistance(currentState, successorPos, goalPoint) + (-100) * min(dists)
                    else:
                        fx = self.getDistance(currentState, successorPos, goalPoint) + (-100) * 10
                else:
                    fx = self.getDistance(currentState, successorPos, goalPoint) + (-100) * 10
                    # self.ischased   =   False

                ## the item pushed into priorityQueue is the successor + past-move ##
                item = [successor, beforeAction + [action]]
                exploring.push(item, fx)

    def evaluate(self, currentState, successorState, action):

        qVal = 0
        features = self.getFeatures(currentState, successorState, action)
        weights = self.getWeights(currentState, action)
        qVal += features['closestFood'] * weights['closestFood']
        qVal += features['closestCapsule'] * weights['closestCapsule']
        qVal += features['enemyDistance'] * weights['enemyDistance']
        qVal += features['trapped'] * weights['trapped']
        return qVal

    def getFeatures(self, currentState, successorState, action):
        """
    Places characteristics in counter data.
    Things to calculate every move.
    """
        # Modify values returned by the functions to create a meaning
        # These values get multiplied by the weights to get an evaluation
        # EX) Food is a distance closest to 1. Either find a value for it
        # or check if the action taken gets you closer to the closest food?
        features = util.Counter()
        features['closestFood'] = self.getClosestFood(currentState, successorState, self.average)
        features['closestCapsule'] = self.getClosestCapsule(currentState, successorState)
        features['enemyDistance'] = self.stayAway(currentState, successorState)
        features['trapped'] = self.calculateTrapped(successorState, currentState, action)
        return features

    def getWeights(self, gameState, action):
        """
    Returns a dict of values that deem how important a charateristic is.
    """
        # Holds a dict of weights for features. Change these to reflect importance.
        return {'closestFood': 1.0, 'closestCapsule': 1.0, 'enemyDistance': -2.0, 'trapped': -2.0}

    ####################
    # HELPER FUNCTIONS #
    ####################

    """
  List of Characters to Compute:
  1) Get closest food - closest path to other side
  2) Get Noise - approximate position of enemies
  3) Get Capusle - While invulnerable, eat as much food as can
            while always moving towards noise (chase opponent if found)
  """

    def getClosestFood(self, currentState, successorState, average):
        # CHANGE NUMBERS TO FIT INTO THE FITNESS SECTION
        totalEval = 0
        curState = currentState.getAgentState(self.index)
        currentPos = curState.getPosition()
        succState = successorState.getAgentState(self.index)
        newPos = succState.getPosition()
        currentFood = self.getFood(currentState).asList()
        capsuleList = self.getCapsules(currentState)
        eatingList = currentFood + capsuleList

        tempFood = []
        for food in eatingList:
            if self.index < 2:
                if food[1] >= average:
                    tempFood.append(food)
            else:
                if food[1] < average:
                    tempFood.append(food)

        if len(tempFood) > 0:
            eatingList = tempFood

        # If Pacman can get food in its next state increase priority
        for food in eatingList:
            if newPos == food:
                totalEval = totalEval + 5
        # If Pacman moves towards the closest food deem action valuable
        closestFoodDist = float("inf")
        closestFood = [(0, 0)]
        for food in eatingList:
            currentDist = self.getMazeDistance(currentPos, food)
            if currentDist < closestFoodDist:
                closestFoodDist = currentDist
                closestFood[0] = food
        if self.getMazeDistance(newPos, closestFood[0]) < closestFoodDist:
            totalEval = totalEval + 5
        return totalEval

    # '''
    #  Variables needed for computation
    #  foodList = self.getFood(currentState).asList()
    #  myState = successorState.getAgentState(self.index)
    #  myPos = myState.getPosition()
    #  closestFoodDist = float("inf")
    #  for food in foodList:
    #    if self.getMazeDistance(myPos, food) < closestFoodDist:
    #      closestFoodDist = self.getMazeDistance(myPos, food)
    #  foodFactor = 1.0 - float(closestFoodDist)/75.0
    #  #foodFactor = foodFactor * 1
    #  #print("Food:", foodFactor)
    #  return foodFactor
    # '''

    def getClosestCapsule(self, currentState, successorState):
        # Maybe do same as food and check if moving towards?
        # Or check if move towards if ghost is visible?
        capsules = self.getCapsules(currentState)
        curr = successorState.getAgentState(self.index).getPosition()
        # if capsules is []:
        # return 0
        # print capsules

        minimum_distance = float("inf")
        for capsule in capsules:
            distance = self.getMazeDistance(curr, capsule)
            if distance < minimum_distance:
                minimum_distance = distance

        # print minimum_distance
        if not capsules:
            # print 0
            return 0

        capsuleFactor = 1 - float(minimum_distance) // 100.0
        capsuleFactor = capsuleFactor * 1
        # print ("capsule:", capsuleFactor)
        return capsuleFactor

    def getApproximateEnemy(self, gameState, index):
        """#####################################
        Edited Code:
            My plan is to check whether the next move is going to be
            trapped or not. If it's trapped and noise is less than 4,
            (4 is just my estimate distance) then it will return -inf
            because you will die automatically if you go into that trapped
            area. The trapped function is at "calculateTrapped" function.
    """
        values = gameState.getAgentDistances()
        # print values
        if gameState.isOnRedTeam(self.index):
            # print values[self.index]
            return values[self.index]
        else:
            # print values[self.index-1]
            return values[self.index - 1]

        # print index, values[index]
        # return values[index]

    def getEnemyDistance(self, currentState, successorState, action):
        distance_away_from_enemy = self.getApproximateEnemy(successorState, self.index)
        if 6 >= distance_away_from_enemy >= -6:
            return float("-inf")
        elif 9 >= distance_away_from_enemy >= -3:
            if self.calculateTrapped(successorState, currentState, action):
                return float("-inf")
        return distance_away_from_enemy

    def stayAway(self, currentState, successorState):
        currentPos = currentState.getAgentState(self.index).getPosition()
        newPos = successorState.getAgentState(self.index).getPosition()
        enemyPositions = self.newEnemyDistance(currentState, successorState)
        opponents = self.getOpponents(currentState)
        if len(enemyPositions) == 0:
            return 0
        else:
            for opponent in opponents:
                if not currentState.getAgentState(opponent).isPacman:
                    if currentState.getAgentState(opponent).scaredTimer > 0:
                        return 0
            if currentState.getAgentState(self.index).isPacman:
                for enemy in enemyPositions:
                    if self.getMazeDistance(currentPos, enemy) < 6:
                        factor = 1.0 - float(self.getMazeDistance(currentPos, enemy)) // 75.0
                        return factor
                    # if abs(self.getMazeDistance(newPos, enemy) - self.getMazeDistance(currentPos, enemy)) < 3:
                    # return self.getMazeDistance(newPos, enemy)
        # print("How did I get here?")
        return 0

    def newEnemyDistance(self, currentState, successorState):
        opponents = self.getOpponents(currentState)
        enemyPositions = []
        for opponent in opponents:
            enemyPosition = successorState.getAgentState(opponent).getPosition()
            if enemyPosition is not None:
                enemyPositions.append(enemyPosition)
        return enemyPositions

    def calculateTrapped(self, successorState, currentState, action):
        """
            To know whether you are going to be trapped or not,
            I said that if possible actions in successorState has
            one legal action and taking that action brings back to
            the currentState position, then you know you are trapped.
            However, this doesnt calculate if the trapped space is
            bigger, not just one spot.
      """
        trapped = self.goThroughSuccessorForTrap(successorState, action, count=5)
        if trapped == True:
            enemyPositions = self.newEnemyDistance(currentState, successorState)
            opponents = self.getOpponents(currentState)
            currentPos = currentState.getAgentState(self.index).getPosition()
            for enemy in enemyPositions:
                if self.getMazeDistance(currentPos, enemy) < 6:
                    return 10
        return 0

    def goThroughSuccessorForTrap(self, successorState, prevAction, count):
        """
            prevAction is a previous action that it took to create successorState.
            This action is passed down from the chooseAction function from the
            way beginning. Count is basically how much are you willing to travel
            deep down to check whether if it's trapped or not.
      """
        if count >= 1:
            actions = successorState.getLegalActions(self.index)
            actions.remove('Stop')
            converted_action = 'Stop'
            """
                Here, if prevAction was South, that means the possible action
                in successorState must contain opposite direction of prevAction.
                I am going to remove the opposite direction, which is North from
                actions in successorState. Now, if there are no actions to take,
                it means that you're trapped. But if there is 1 legal action, it
                is the case where the trap space is a long hallway. I said count
                equal to 4, hoping that the hallway is not longer than 4.
                If there are more than 1 actions even after deleting the opposite
                direction of previous action, that means you are heading into a
                2D space, which pacman can run away from the ghost even if it takes
                the path that is surrounded by wall.
                This function still returns true or false if its trapped or not.
                Problem: when there is one trap space like below drawing, pacman
                still takes that trap space. I think it has to do with the
                noise function. Might want to fix the constraint.
                __   __
                  |_|
          """

            if prevAction == 'North':
                converted_action = 'South'
            elif prevAction == 'South':
                converted_action = 'North'
            elif prevAction == 'East':
                converted_action = 'West'
            elif prevAction == 'West':
                converted_action = 'East'

            if converted_action in actions:
                actions.remove(converted_action)
            if len(actions) == 0:
                position = successorState.getAgentState(self.index).getPosition()
                # print "trapped place found!", self.index, position
                # print "trapped found near 5 distance"
                return True
            elif len(actions) == 1:
                next_successor_state = successorState.generateSuccessor(self.index, actions[0])
                return self.goThroughSuccessorForTrap(next_successor_state, actions[0], count - 1)
            else:
                return False
        else:
            return False


class DefensiveAgent(ReflexCaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # Get distance
        self.distancer.getMazeDistances()
        # Set defensive area
        self.setDefensiveArea(gameState)

        self.catchState = False

        self.start = gameState.getAgentPosition(self.index)

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.coreDefendingArea = []
        self.target = None
        self.remainFoodList = []
        self.isFoodEaten = False
        self.patrolDict = {}
        self.tick = 0
        self.gazeboDict = {}
        self.catchState = False

    def getLayoutInfo(self, gameState):
        layoutInfo = []
        layoutWidth = gameState.data.layout.width
        layoutHeight = gameState.data.layout.height
        layoutCentralX = (layoutWidth - 2) // 2
        if not self.red:
            layoutCentralX += 1
        layoutCentralY = (layoutHeight - 2) // 2
        layoutInfo.extend((layoutWidth, layoutHeight, layoutCentralX, layoutCentralY))
        return layoutInfo

    def setDefensiveArea(self, gameState):

        layoutInfo = self.getLayoutInfo(gameState)

        for i in range(1, layoutInfo[1] - 1):
            if not gameState.hasWall(layoutInfo[2], i):
                self.coreDefendingArea.append((layoutInfo[2], i))

        desiredSize = layoutInfo[3]
        currentSize = len(self.coreDefendingArea)

        while desiredSize < currentSize:
            self.coreDefendingArea.remove(self.coreDefendingArea[0])
            self.coreDefendingArea.remove(self.coreDefendingArea[-1])
            currentSize = len(self.coreDefendingArea)

        while len(self.coreDefendingArea) > 2:
            self.coreDefendingArea.remove(self.coreDefendingArea[0])
            self.coreDefendingArea.remove(self.coreDefendingArea[-1])

    def ForcedDefend(self, gameState):
        candidateActions = []
        actions = gameState.getLegalActions(self.index)
        reversed_direction = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        actions.remove(Directions.STOP)
        if reversed_direction in actions:
            actions.remove(reversed_direction)

        for a in actions:
            new_state = gameState.generateSuccessor(self.index, a)
            if not new_state.getAgentState(self.index).isPacman:
                candidateActions.append(a)

        if len(candidateActions) == 0:
            self.tick = 0
        else:
            self.tick = self.tick + 1

        if self.tick > 20 or self.tick == 0:
            candidateActions.append(reversed_direction)

        return candidateActions

    def chooseAction(self, gameState):
        # Our home food list
        ourCurrentFoodList = self.getFoodYouAreDefending(gameState).asList()
        # our position
        myPos = gameState.getAgentPosition(self.index)
        if myPos == self.target:
            self.target = None
        # Get the cloest invader's position and set target as invader
        opponentsIndices = []
        threateningInvaderPos = []
        cloestInvaders = []
        # set distance to infinity
        minDistance = 99999

        myScore = self.getScore(gameState)
        myState = gameState.getAgentState(self.index)

        opponentsIndices = self.getOpponents(gameState)
        for opponentIndex in opponentsIndices:
            opponent = gameState.getAgentState(opponentIndex)
            # opponent are eating our food
            if opponent.isPacman and opponent.getPosition() is not None:
                opponentPos = opponent.getPosition()
                threateningInvaderPos.append(opponentPos)

        # When opponent pacman can be eaten by us
        if len(threateningInvaderPos) > 0:
            for position in threateningInvaderPos:
                distance = self.getMazeDistance(position, myPos)
                if distance < minDistance:
                    minDistance = distance
                    cloestInvaders.append(position)
            self.target = cloestInvaders[-1]

        # get the eaten food position
        else:
            if len(self.remainFoodList) > 0 and len(ourCurrentFoodList) < len(self.remainFoodList):
                eatenFood = set(self.remainFoodList) - set(ourCurrentFoodList)

                self.target = eatenFood.pop()

        self.remainFoodList = ourCurrentFoodList

        if self.target is None:
            if len(ourCurrentFoodList) <= 4:
                highPriorityFood = ourCurrentFoodList + self.getCapsulesYouAreDefending(gameState)
                self.target = random.choice(highPriorityFood)
            else:
                self.target = random.choice(self.coreDefendingArea)
        # evaluates candiateActions and get the best
        candidateActions = self.ForcedDefend(gameState)
        goodActions = []
        fValues = []

        for a in candidateActions:
            new_state = gameState.generateSuccessor(self.index, a)
            newPos = new_state.getAgentPosition(self.index)
            goodActions.append(a)
            fValues.append(self.getMazeDistance(newPos, self.target))

        best = min(fValues)
        bestActions = [a for a, v in zip(goodActions, fValues) if v == best]
        bestAction = random.choice(bestActions)

        return bestAction
