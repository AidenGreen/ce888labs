# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a 
# state.GetRandomMove() or state.DoRandomRollout() function.
# 
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in 
# the UCTPlayGame() function at the bottom of the code.
# 
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
# 
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

from math import *


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

import random
import pandas as pd
import numpy as np
import random
from sklearn.externals import joblib


class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic 
        zero-sum game, although they can be enhanced and made quicker, for example by using a 
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """
    def __init__(self):
            self.playerJustMoved = 2 # At the root pretend the player just moved is player 2 - player 1 has the first move
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        self.playerJustMoved = 3 - self.playerJustMoved
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass

class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """
    def __init__(self):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0,0,0,0,0,0,0,0,0] # 0 = empty, 1 = player 1, 2 = player 2
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] == self.board[y] == self.board[z] and self.board[x] != 0:
                if self.board[x] == playerjm:
                    return 1
                else:
                    return 0
        # if self.GetMoves() == []: return 0.5 # draw
        return 0.5 # draw
        assert False # Should not be possible to get here

    def __repr__(self):
        s= ""
        for i in range(9): 
            s += ".XO"[self.board[i]]
            if i % 3 == 2: s += "\n"
        return s

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later
        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        reward = result

        # if result == self.playerJustMoved:
        #     reward = 1
        # elif result == 0:
        #     reward = 0.5
        # else:
        #     reward = 0

        self.visits += 1
        self.wins += reward

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []: # while state is non-terminal
            
            movement = TreePredicter([state.board])
            ram = random.randrange(1,10)
            ram = float(ram/10)
            # state.DoMove(random.choice(state.GetMoves()))
            if ram > 0.1:
                state.DoMove(movement)
            else:
                state.DoMove(random.choice(state.GetMoves()))
            
            if state.GetResult(node.playerJustMoved) != 0.5:
                # print("Arrive")
                break

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    #if (verbose): print(rootnode.TreeToString(0)) 
    #else: print(rootnode.ChildrenToString()) 

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited
                
def UCTPlayGame():

    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    state = OXOState() # uncomment to play OXO

    moveRecord =[0,0,0,0,0,0,0,0,0,0]
    stateRecord =[]
    i = 0   
    result = 0
    while (state.GetMoves() != []):
        
        #print(str(state)) 
        if state.playerJustMoved == 1:
            m = UCT(rootstate = state, itermax = 1000, verbose = False) # play with values for itermax and verbose = True
        else:
            m = UCT(rootstate = state, itermax = 100, verbose = False)
        #print("Best Move: " + str(m) + "\n") 
        
        temp = state.board.copy()
        state.DoMove(m)
        stateRecord.append(temp)
        moveRecord[i] = m
        i += 1
        result = state.GetResult(state.playerJustMoved)
        # early stop
        if result != 0.5:
            break
    
    
    if result != 0.5:
        print("Player " + str(state.playerJustMoved) + " wins!")
        moveRecord[9] = state.playerJustMoved
    else: 
        print("Nobody wins!") 
        state.playerJustMoved =2.0
        moveRecord[9] = 0
    
    #Add the winner at the end
    #moveRecord.append(int(state.playerJustMoved))
    print(state)

    return moveRecord,stateRecord
 

def DecisionTree_Classiffier(*data):

    x_train, x_test, y_train, y_test = data

    #param_distribs = {'max_features':['auto'],'min_weight_fraction_leaf':[0,0.25,0.5],'criterion':['entropy','gini']}
    #1,0,100
    #-------------------------------------------   
    clf= DecisionTreeClassifier(

        min_samples_leaf  = 1,
        min_impurity_decrease = 0,
        max_depth = 100,
        max_features ='auto',
        criterion = 'entropy'
        
        )
    try:
        clf = joblib.load("TreeModel.m")
        joblib.dump(clf, "TreeModel_O.m")
        pass
    except:
        pass
    clf.fit(x_train,y_train)
    joblib.dump(clf, "TreeModel.m")
    print("Train score:%s"%(clf.score(x_train,y_train)))
    print("Test score:%s"%(clf.score(x_test,y_test)))

    # random_search = RandomizedSearchCV(clf,param_distribs,n_iter=50,cv=3)

    # random_search.fit(x_train,y_train)

    # print("RS Train score:%s"%(random_search.score(x_train,y_train)))
    # print("RS Test score:%s"%(random_search.score(x_test,y_test)))

    # print(random_search.best_params_)

    #-------------------------------------------   
    # clf.fit(x_train,y_train)
    # print("Train score:%s"%(clf.score(x_train,y_train)))
    # print("Test score:%s"%(clf.score(x_test,y_test)))

def load_data(moveRecords):
    inputData = moveRecords[:,:-1]
    outputData = moveRecords[:,-1]

    return train_test_split(inputData,outputData,test_size=0.1,random_state=0)

def readFile(path):

    openfile=open(path,'r',encoding="utf-8")

    lines = openfile.readlines()
    data = []
    for line in lines:
        line = line.replace('\n','')
        if line != '':
            temp = line.split(',')
            data.append(temp)
    return data

def TreePredicter(curState,version = "New"):
    clf= DecisionTreeClassifier(

        min_samples_leaf  = 1,
        min_impurity_decrease = 0,
        max_depth = 100,
        max_features ='auto',
        criterion = 'entropy'
        
        )
    if(version == "New"):
        clf = joblib.load("TreeModel.m")
    else:
        clf = joblib.load("TreeModel_O.m")
    output = clf.predict(curState)
    output = int(output[0])
    state = curState[0]
    if state[output] != 0:
        pos = []
        for i in range(9):
            if state[i] == 0:
                pos.append(i)
        output = random.choice(pos) 
    return output

def CalResult(state):
    """ Get the game result from the viewpoint of playerjm. """
    state=state[0]
    for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
        if state[x] == state[y] == state[z]:
            return state[x]
    return 0


if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """
    moveRecords =[]
    stateRecord =[]
    trainData = []

    results = [0,0,0]
    inputFile = readFile('data.csv')
    trainData = inputFile
    print(len(trainData))

    array=np.array(trainData)
    x_train,x_test,y_train,y_test=load_data(array)
    DecisionTree_Classiffier(x_train,x_test,y_train,y_test)

    k = 0
    while True:
        k = k +1
        if k >= 11:
            print("Stop!")
        for i in range(10):
            print("Epoch:",i,"Round:",k)
            moveRecord,stateRecord = UCTPlayGame()
            for j in range(len(stateRecord)):
                if moveRecord[9] != 0:
                    temp = stateRecord[j].copy()
                    temp.append(moveRecord[j])
                    trainData.append(temp.copy())
        

        random.shuffle(trainData)
        
        # trainSet = list(set([tuple(t) for t in trainData]))
        # array=np.array(trainSet)

        # 360
        array=np.array(trainData)
        x_train,x_test,y_train,y_test=load_data(array)
        DecisionTree_Classiffier(x_train,x_test,y_train,y_test)
        data = pd.DataFrame(data=array)
        data.to_csv("data.csv",index = False,header=False)
        
        for i in range(10):
            state = [[0,0,0,0,0,0,0,0,0]]
            p_1 = random.choice([0,1,2,3,4,5,6,7,8])
            state[0][p_1] = 1
            for j in range(4):
                p_2 = TreePredicter(state,"Old")
                state[0][p_2] = 2
                p_1 = TreePredicter(state)
                state[0][p_1] = 1
            result = CalResult(state)
            results[result] += 1

        print(results)
        

