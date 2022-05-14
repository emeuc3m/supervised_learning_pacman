from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import copy
import util
from game import Actions
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
from wekaI import Weka


class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        '''NEW'''
        self.print_data = ""
        self.chosen_action = None
        self.future_score = []

        self.weka = Weka()
        self.weka.start_jvm()


    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True


    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState


    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        '''NEW'''
        if gameState.isWin():
            self.printLineData(gameState, 'Stop')
            #stop de java machine
            self.weka.stop_jvm()
            return 'Stop'

        action = self.chooseAction(gameState) 
        self.printLineData(gameState, action)
        return action

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

    def __load_attributes(self):

        self.print_data = ("@RELATION pacman-training\n\n" + 
                "@ATTRIBUTE pacman_x NUMERIC\n" + #CURRENT DATA
                "@ATTRIBUTE pacman_y NUMERIC\n" + 
                "@ATTRIBUTE living_ghosts NUMERIC\n" +
                "@ATTRIBUTE num_food NUMERIC\n" +
                "@ATTRIBUTE move_west NUMERIC\n" +
                "@ATTRIBUTE move_east NUMERIC\n" +
                "@ATTRIBUTE move_north NUMERIC\n" +
                "@ATTRIBUTE move_south NUMERIC\n" +
                "@ATTRIBUTE move_stop NUMERIC\n" +
                "@ATTRIBUTE ghost1_x NUMERIC\n" +
                "@ATTRIBUTE ghost1_y NUMERIC\n" +
                "@ATTRIBUTE ghost1_distance NUMERIC\n" +
                "@ATTRIBUTE ghost1_alive NUMERIC\n" +
                "@ATTRIBUTE ghost2_x NUMERIC\n" +
                "@ATTRIBUTE ghost2_y NUMERIC\n" +
                "@ATTRIBUTE ghost2_distance NUMERIC\n" +
                "@ATTRIBUTE ghost2_alive NUMERIC\n" +
                "@ATTRIBUTE ghost3_x NUMERIC\n" +
                "@ATTRIBUTE ghost3_y NUMERIC\n" +
                "@ATTRIBUTE ghost3_distance NUMERIC\n" +
                "@ATTRIBUTE ghost3_alive NUMERIC\n" +
                "@ATTRIBUTE ghost4_x NUMERIC\n" +
                "@ATTRIBUTE ghost4_y NUMERIC\n" +
                "@ATTRIBUTE ghost4_distance NUMERIC\n" +
                "@ATTRIBUTE ghost4_alive NUMERIC\n" +
                "@ATTRIBUTE closest_food NUMERIC\n" +
                "@ATTRIBUTE score_now NUMERIC\n" +
                "@ATTRIBUTE living_ghosts_after NUMERIC\n" + #FUTURE DATA
                "@ATTRIBUTE num_food_after NUMERIC\n" +
                "@ATTRIBUTE ghost1_distance_after NUMERIC\n" +
                "@ATTRIBUTE ghost1_alive_after NUMERIC\n" +
                "@ATTRIBUTE ghost2_distance_after NUMERIC\n" +
                "@ATTRIBUTE ghost2_alive_after NUMERIC\n" +
                "@ATTRIBUTE ghost3_distance_after NUMERIC\n" +
                "@ATTRIBUTE ghost3_alive_after NUMERIC\n" +
                "@ATTRIBUTE ghost4_distance_after NUMERIC\n" +
                "@ATTRIBUTE ghost4_alive_after NUMERIC\n" +
                "@ATTRIBUTE closest_food_after NUMERIC\n" +
                "@ATTRIBUTE score_after NUMERIC\n\n" +
                "@ATTRIBUTE action_taken {West, East, North, South, Stop}\n\n" + #ACTION THAT TOOK THE PACMAN TO THE FUTURE STATE
                "@DATA\n"
            )

    def load_data(self, gameState, current_score):

        self.print_data = (str(gameState.getPacmanPosition()[0]) + ', ' + #pacman position x
            str(gameState.getPacmanPosition()[1]) + ', ' + #pacman position y
            str(self.count_living_ghosts(gameState)) + ', ' + #living ghosts
            str(gameState.getNumFood()) + ', ' + #number of food left
            str(int("West" in gameState.getLegalActions())) + ', ' +  #west is legal action?
            str(int("East" in gameState.getLegalActions())) + ', ' +  #east is legal action?
            str(int("North" in gameState.getLegalActions())) + ', ' +  #north is legal action?
            str(int("South" in gameState.getLegalActions())) + ', ' +  #south is legal action?
            str(int("Stop" in gameState.getLegalActions())) + ', ' +  #stop is legal action?
            str(gameState.getGhostPositions()[0][0]) + ', ' + #ghost1 position x
            str(gameState.getGhostPositions()[0][1]) + ', ' + #ghost1 position y
            str(gameState.data.ghostDistances[0] if gameState.data.ghostDistances[0] != None else 0) + ', ' + #ghost1 distance
            str(int(gameState.getLivingGhosts()[1])) + ', ' + #ghost1 alive
            str(gameState.getGhostPositions()[1][0]) + ', ' + #ghost2 position x
            str(gameState.getGhostPositions()[1][1]) + ', ' + #ghost2 position y
            str(gameState.data.ghostDistances[1] if gameState.data.ghostDistances[1] != None else 0) + ', ' + #ghost2 distance
            str(int(gameState.getLivingGhosts()[2])) + ', ' + #ghost2 alive
            str(gameState.getGhostPositions()[2][0]) + ', ' + #ghost3 position x
            str(gameState.getGhostPositions()[2][1]) + ', ' + #ghost3 position y
            str(gameState.data.ghostDistances[2] if gameState.data.ghostDistances[2] != None else 0) + ', ' + #ghost3 distance
            str(int(gameState.getLivingGhosts()[3])) + ', ' + #ghost3 alive
            str(gameState.getGhostPositions()[3][0]) + ', ' + #ghost4 position x
            str(gameState.getGhostPositions()[3][1]) + ', ' + #ghost4 position y
            str(gameState.data.ghostDistances[3] if gameState.data.ghostDistances[3] != None else 0) + ', ' + #ghost4 distance
            str(int(gameState.getLivingGhosts()[4])) + ', ' + #ghost4 alive
            str(gameState.getDistanceNearestFood() if gameState.getDistanceNearestFood()!= None else 0) + ', ' + #distance closest food
            str(current_score) + ', ' #current score
        )

        

    def count_living_ghosts(self, gameState):

        count = 0
        for life in gameState.getLivingGhosts():
            if life == True:
                count += 1

        return count


    '''NEW
    Function to save data in .arff format'''
    def printLineData(self, gameState, action):
        
        #only useful when there are exactly 4 ghosts

        self.future_score.append(gameState.getScore())
        
        try:

            file = open("all_data_pacman.arff", 'r')
            file.close()

        except:

            file = open("all_data_pacman.arff", 'w')
            self.__load_attributes()
            file.write(self.print_data)
            file.close()

            self.print_data = ""

        #avoid queing delays through loop
        while self.future_score != []:
            #@data to be stored in .arff format file
            if self.print_data == "":

                #save chosen action
                self.chosen_action = action

                #current score
                current_score = self.future_score.pop(0)

                #situation in current tick
                self.load_data(gameState, current_score)

                #delay 1 tick
                if current_score == 0:
                    return

            else:

                
                #add future score data

                self.print_data += (str(self.count_living_ghosts(gameState)) + ', ' + #future num of ghosts
                    str(gameState.getNumFood()) + ', ' + #future number of food left
                    str(gameState.data.ghostDistances[0] if gameState.data.ghostDistances[0] != None else 0) + ', ' + #future ghost1 distance
                    str(int(gameState.getLivingGhosts()[1])) + ', ' + #future ghost1 alive
                    str(gameState.data.ghostDistances[1] if gameState.data.ghostDistances[1] != None else 0) + ', ' + #future ghost2 distance
                    str(int(gameState.getLivingGhosts()[2])) + ', ' + #future ghost2 alive
                    str(gameState.data.ghostDistances[2] if gameState.data.ghostDistances[2] != None else 0) + ', ' + #future ghost3 distance
                    str(int(gameState.getLivingGhosts()[3])) + ', ' + #future ghost3 alive
                    str(gameState.data.ghostDistances[3] if gameState.data.ghostDistances[3] != None else 0) + ', ' + #future ghost4 distance
                    str(int(gameState.getLivingGhosts()[4])) + ', ' + #future ghost4 alive
                    str(gameState.getDistanceNearestFood() if gameState.getDistanceNearestFood()!= None else 0) + ', ' + #future distance closest food
                    str(self.future_score[0]) + ', ' + #future score
                    str(self.chosen_action) + '\n' #chosen action 
                )

                # write data in output .arff file
                file = open("all_data_pacman.arff", 'a')
                file.write(self.print_data)
                file.close()

                #reboot data
                self.print_data = ""
               


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

'''Added manhattanDistance'''
from distanceCalculator import Distancer, manhattanDistance
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())
        
    
    def chooseAction(self, gameState):

        #self.countActions = self.countActions + 1        
        #move = Directions.STOP

        living_ghosts = gameState.getLivingGhosts() # Array of bools
        ghost_positions = gameState.getGhostPositions()
        living_ghosts_positions = []
        
        for ii, alive in enumerate(living_ghosts):
            if ii != 0: # Because pacman is always the agent at pos 0 (for some reason)
                if alive:
                    living_ghosts_positions.append(ghost_positions[ii-1])

        #print(living_ghosts_positions)
        
        pacman_position = gameState.getPacmanPosition()


        open_list = []
        closed_list = []
        walls = gameState.getWalls()
        ghost_distances = [util.manhattanDistance(pacman_position, x) for x in living_ghosts_positions]
        

        # Nodes have this format: [pacman_xy, direction_list, h_value]
        start = [pacman_position, [], min(ghost_distances)]
        
        #initialize open list
        open_list.append(start)

        # While there are nodes on the open list
        while open_list:

            # Create the children
            living_ghosts_positions = []
            for ii, alive in enumerate(living_ghosts):
                if ii != 0: # Because pacman is always the agent at pos 0 (for some reason)
                    if alive:
                        living_ghosts_positions.append(ghost_positions[ii-1]) 


            current = open_list[0]

            current_index = 0


            # Get the smallest h value node.
            for ii, open_node in enumerate(open_list):

                if open_node[2] < current[2]: 

                    current = open_node
                    current_index = ii

            #take out the node with the lowest h value and place it in closed list
            closed_list.append(open_list.pop(current_index))

            # If pacman's position is the same as a ghost's, that state is a solution
            if current[0] in living_ghosts_positions:
                
                return current[1][0]


            # Expand the node
            neighbors = Actions.getLegalNeighbors(current[0], walls) # (xy_pos, dir)
            children = [] 

            for node in neighbors:     
                ghost_distances = [util.manhattanDistance(current[0], x) for x in living_ghosts_positions]
                actions = copy.copy(current[1])
                actions.append(copy.copy(node[1]))
                child = [node[0], actions, min(ghost_distances)]
                children.append(child)
          
            for child in children:
            # If child node was already explored, ignore it
                closed = False
                for node in closed_list:
                    if child[0] == node[0]:
                        closed = True
                        break
                if closed:
                    continue
                valid = True
                for open_child in open_list:
                    # If we already found the same node but with lesser cost, skip this one
                    if child[0] == open_child[0] and child[2] >= open_child[2]:
                        valid = False
                
                if valid:
                    open_list.append(child)
            
        return 'Stop'


class WekaBustersAgent(BustersAgent):

    "Buster Agent built from weka model"

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)


    def chooseAction(self, gameState):

        value = [gameState.getPacmanPosition()[0],  #pacman position x
            gameState.getPacmanPosition()[1],  #pacman position y
            self.count_living_ghosts(gameState),  #living ghosts
            gameState.getNumFood(), #number of food left
            int("West" in gameState.getLegalActions()),   #west is legal action?
            int("East" in gameState.getLegalActions()),   #east is legal action?
            int("North" in gameState.getLegalActions()),   #north is legal action?
            int("South" in gameState.getLegalActions()),   #south is legal action?
            gameState.getGhostPositions()[0][0],  #ghost1 position x
            gameState.getGhostPositions()[0][1],  #ghost1 position y
            gameState.data.ghostDistances[0] if gameState.data.ghostDistances[0] != None else 0,  #ghost1 distance
            int(gameState.getLivingGhosts()[1]),  #ghost1 alive
            gameState.getGhostPositions()[1][0],  #ghost2 position x
            gameState.getGhostPositions()[1][1],  #ghost2 position y
            gameState.data.ghostDistances[1] if gameState.data.ghostDistances[1] != None else 0,  #ghost2 distance
            int(gameState.getLivingGhosts()[2]),  #ghost2 alive
            gameState.getGhostPositions()[2][0], #ghost3 position x
            gameState.getGhostPositions()[2][1],  #ghost3 position y
            gameState.data.ghostDistances[2] if gameState.data.ghostDistances[2] != None else 0,  #ghost3 distance
            int(gameState.getLivingGhosts()[3]),  #ghost3 alive
            gameState.getGhostPositions()[3][0], #ghost4 position x
            gameState.getGhostPositions()[3][1],  #ghost4 position y
            gameState.data.ghostDistances[3] if gameState.data.ghostDistances[3] != None else 0,  #ghost4 distance
            int(gameState.getLivingGhosts()[4]),  #ghost4 alive
            gameState.getDistanceNearestFood() if gameState.getDistanceNearestFood()!= None else 0,  #distance closest food
            gameState.getScore()]  #current score

        #print(value)
        chosen_action = self.weka.predict("../weka_models/RandomForestOld.model", value, "../arff_files/fase2_clasificacion/training_tutorial1_v5.arff")
        
        #prevent choosing illegal actions
        if chosen_action not in gameState.getLegalActions():
            
            #select random legal action
            return gameState.getLegalActions()[random.randint(0,len(gameState.getLegalActions())-1)]

        else:

            return chosen_action
