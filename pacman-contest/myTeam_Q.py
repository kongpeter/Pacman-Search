from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions,Actions
import game
from util import nearestPoint
import layout

'''
Team Creation:
(1) Offensive Agent
(2) Defensive Agent
'''


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


'''
Define Reflex Agent extends from capture Agent
'''


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


'''
Offensive Agent:
(1) Become pacman to eat opponent food
'''


class OffensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.start = gameState.getAgentPosition(self.index)
        self.catchState = False
        self.epsilon = 0.0  # exploration prob
        self.alpha = 0.5  # learning rate
        self.discountRate = 0.8
        self.weights = {'closest-food': -1.054855704637854,
                        'bias': 1.2970278626350917,
                        '#-of-ghosts-1-step-away': -0.46065450914953676,
                        'successorScore': 0.27085439967059727,
                        'eats-food': 2.6441982416502645}
        """
        Open weights file if it exists, otherwise start with empty weights.
        NEEDS TO BE CHANGED BEFORE SUBMISSION
        try:
            with open('weights.txt', "r") as file:
                self.weights = eval(file.read())
        except IOError:
                return
        """
    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        return features * self.weights

    def getValue(self, gameState):
        qVals = []
        legalActions = gameState.getLegalActions(self.index)
        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                qVals.append(self.getQValue(gameState, action))
            return max(qVals)

    def getPolicy(self, gameState):
        values = []
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove(Directions.STOP)
        if len(legalActions) == 0:
            return None
        else:
            for action in legalActions:
                # self.updateWeights(gameState, action)
                values.append((self.getQValue(gameState, action), action))
        return max(values)[1]

    def chooseAction(self, gameState):
        # When pacman was eaten by opponent
        opponentCapsule = self.getCapsules(gameState)
        myState = gameState.getAgentState(self.index)
        legalActions = gameState.getLegalActions(self.index)
        action = None

        if gameState.getAgentPosition(self.index) == self.start:
            self.catchState = False

        if self.catchState or myState.numCarrying >= 5:
            goalPoint = self.start
            action = self.aStarSearch(gameState, goalPoint)
        else:
            if len(legalActions) != 0:
                prob = util.flipCoin(self.epsilon)
                if prob:
                    action = random.choice(legalActions)
                else:
                    action = self.getPolicy(gameState)


        self.checkCatch(gameState.generateSuccessor(self.index, action))

        return action

    def getDistance(self, gameState, successorPos, goalPoint):
        minDistance = 9999
        if goalPoint is None:
            foodList = self.getFood(gameState).asList()
            capsuleList = self.getCapsules(gameState)
            eatingList = foodList + capsuleList

            if successorPos in eatingList:
                # distance reward is -100
                distance = 0 + (-500)
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

        if len(opponentGhost) > 0 and minDistance <= 3:
            # ghost scared:
            if opponentGhost[-1].scaredTimer > 0:
                self.catchState = False
            else:
                self.catchState = True
        else:
            self.catchState = False

    def aStarSearch(self, gameState, goalPoint):

        # initialization
        explored = []
        exploring = util.PriorityQueue()
        loopState = False
        exploring.push([gameState, []], 0)
        food_list = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        eatList = food_list + capsuleList

        while not loopState:

            popItem = exploring.pop()
            currentState = popItem[0]
            beforeAction = popItem[1]
            currentPos = currentState.getAgentPosition(self.index)

            # find food and not being catch
            # if currentPos in capsulePosition:
            #     return beforeAction[0]

            if currentPos == goalPoint or (currentPos in eatList and not self.catchState):
                loopState = True
                return beforeAction[0]

            # avoid duplicate explore
            if currentPos in explored:
                continue
            else:
                explored.append(currentPos)
                legalAction = currentState.getLegalActions(self.index)

            for action in legalAction:
                successor = currentState.generateSuccessor(self.index, action)
                successorPosition = successor.getAgentPosition(self.index)
                # ghost position
                opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
                opponentGhosts = [opponent for opponent in opponents if not opponent.isPacman
                                  and opponent.getPosition() is not None]

                if len(opponentGhosts) > 0:
                    # The distance to component ghost
                    distanceToGhost = [self.getMazeDistance(successorPosition, a.getPosition()) for a in opponentGhosts]

                    if min(distanceToGhost) < 4:
                        function = self.getDistance(currentState, successorPosition, goalPoint) + (-10) * min(
                            distanceToGhost)
                    else:
                        function = self.getDistance(currentState, successorPosition, goalPoint) + (-10) * 10
                else:
                    function = self.getDistance(currentState, successorPosition, goalPoint) + (-10) * 10

                item = [successor, beforeAction + [action]]
                exploring.push(item, function)

 # Define features to use. NEEDS WORK
    def getFeatures(self, gameState, action):
        # Extract the grid of food and wall locations
        if self.red:
            food = gameState.getBlueFood()
        else:
            food = gameState.getRedFood()

        walls = gameState.getWalls()
        ghosts = []
        opAgents = CaptureAgent.getOpponents(self, gameState)
        # Get ghost locations and states if observable
        if opAgents:
            for opponent in opAgents:
                opPos = gameState.getAgentPosition(opponent)
                opIsPacman = gameState.getAgentState(opponent).isPacman
                if opPos and not opIsPacman:
                    ghosts.append(opPos)

        # Initialize features
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # Successor Score
        features['successorScore'] = self.getScore(successor)

        # Bias
        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = gameState.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # Number of Ghosts scared
        # features['#-of-scared-ghosts'] = sum(gameState.getAgentState(opponent).scaredTimer != 0 for opponent in opAgents)

        # Closest food
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        # Normalize and return
        features.divideAll(10.0)
        return features

    """
    Iterate through all features and for each feature, update
    its weight values using the following formula:
    w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
    """

    def updateWeights(self, gameState, action):
        features = self.getFeatures(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        # Calculate the reward. NEEDS WORK
        reward = nextState.getScore() - gameState.getScore()

        for feature in features:
            correction = (reward + self.discountRate * self.getValue(nextState)) - self.getQValue(gameState, action)
            self.weights[feature] = self.weights[feature] + self.alpha * correction * features[feature]

    # -------------------------------- Helper Functions ----------------------------------

    # Finds the next successor which is a grid position (location tuple).
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def closestFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no food found
        return None
'''
Defensive Agent
(1) Keep walking in the center of the map
(2) Will track opponent position
'''


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
