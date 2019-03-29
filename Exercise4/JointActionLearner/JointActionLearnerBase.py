#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import itertools
import argparse
import numpy as np
		
class JointQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
		super(JointQLearningAgent, self).__init__()	
		self.numActions = len(self.possibleActions)
		self.learningRatePassed = learningRate
		self.learningRate = 0.0
		self.discountFactor = discountFactor
		self.epsilonPassed = epsilon
		self.epsilon = 0.0
		self.numTeammates = numTeammates # now numTeammates is 1

		self.states = [(x,y) for x in range(5) for y in range(5)] # 生成所有的state
		self.specialStates = []
		self.specialStates.append("GOAL")
		self.specialStates.append("OUT_OF_BOUNDS")
		self.specialStates.append("OUT_OF_TIME")
		# generate the Q-value table
		self.qValue = {}
		for action_1 in self.possibleActions:
			for action_2 in self.possibleActions:
				for state_1 in self.states:
					for state_2 in self.states:
						self.qValue[((state_1,state_2),(action_1,action_2))] =initVals
				for speicalState in self.specialStates:
					self.qValue[(speicalState,(action_1,action_2))] = initVals
		
		# generate the opponent model
		self.oppModel = {} # in fact there is only 1 oppmodel as the number of teammates is 1 in this question. the value of keys is a list: [[probability],[frequence]]
		for state_1 in self.states:
			for state_2 in self.states:
				self.oppModel[(state_1,state_2)] = [[1/self.numActions] * self.numActions,[0]*self.numActions]
		for speicalState in self.specialStates:
				self.oppModel[speicalState] = [[1/self.numActions] * self.numActions,[0]*self.numActions]
		
		self.currentState = []
		self.currentExperience = []
		self.currentEpisode = 0
		self.numTakenActionsLastEpisode = 0

	def setExperience(self, state, act, oppoActions, reward, status, nextState):
		self.currentExperience = (state,act,oppoActions,reward,nextState)
		#raise NotImplementedError
		
	def learn(self):
		state_t = (tuple(self.currentExperience[0][0][0]),tuple(self.currentExperience[0][0][1]))
		action_t = self.currentExperience[1]
		oppoAction_t = self.currentExperience[2][0]
		reward = self.currentExperience[3]
		if self.currentExperience[4] == "GOAL" or self.currentExperience[4] == "OUT_OF_BOUNDS" or self.currentExperience[4]=="OUT_OF_TIME":
			state_next = self.currentExperience[4]
		else:
		    state_next = (tuple(self.currentExperience[4][0][0]),tuple(self.currentExperience[4][0][1]))
		# update the opponent model with new observations.
		oppoaction_index = self.possibleActions.index(oppoAction_t)
		self.oppModel[state_t][1][oppoaction_index]  =  self.oppModel[state_t][1][oppoaction_index] + 1 # calculate total times of this action to be taken in this state.
		countTime_state_t = sum(self.oppModel[state_t][1])
		for action_index in range(self.numActions):
			self.oppModel[state_t][0][action_index] =  self.oppModel[state_t][1][action_index] / countTime_state_t # calculate the pro for each action in this state.
		
		ev_nextState = []
		for action_1 in range(self.numActions):
			ev = 0
			for action_2 in range(self.numActions):
				ev = ev + self.qValue[(state_next,(self.possibleActions[action_1],self.possibleActions[action_2]))] * self.oppModel[state_next][0][action_2]
			ev_nextState.append(ev)
		ev_nextState_max = max(ev_nextState)

		qValue = self.qValue[(state_t,(action_t,oppoAction_t))]
		changeValue =  self.learningRate*(reward+self.discountFactor*(ev_nextState_max)-qValue)
		self.qValue[(state_t,(action_t,oppoAction_t))] =qValue + changeValue
		return changeValue
		raise NotImplementedError

	def act(self):
		# compute the expected value of action a_i in state s against oppmodel
		curstate = (tuple(self.currentState[0][0]),tuple(self.currentState[0][1]))
		ev_curState = []
		for action_1 in range(self.numActions):
			ev = 0
			for action_2 in range(self.numActions):
				ev = ev + self.qValue[(curstate,(self.possibleActions[action_1],self.possibleActions[action_2]))] * self.oppModel[curstate][0][action_2]
			ev_curState.append(ev)
		ev_max = max(ev_curState)
		ev_max_IndexAll = [i for i,j in enumerate(ev_curState) if j==ev_max]
		ev_max_Index = random.choice(ev_max_IndexAll)
		# choose act according to the given state.
		actionIndexAll = [i for i in range(self.numActions)]
		actionIndexAll.remove(ev_max_Index)
		pro = random.random() 
		if pro < self.epsilon:
			# with probability epsilon : choose random act a_i not including the optimal action
			choosedAction = self.possibleActions[random.choice(actionIndexAll)] 
		else:
			# choose best-response act arg max a_i EV(s, a_i)
			choosedAction = self.possibleActions[ev_max_Index]
		return choosedAction
		#raise NotImplementedError

	def setEpsilon(self, epsilon) :
		self.epsilon = epsilon
		#raise NotImplementedError
		
	def setLearningRate(self, learningRate) :
		self.learningRate = learningRate
		#raise NotImplementedError

	def setState(self, state):
		self.currentState = state  # [[player1_loc, player2_loc], defender_loc, ball_loc]
		# [player1_loc, player2_loc] might be [(x1,y1),(x2,y2)] or ['one kind of special state']
		#raise NotImplementedError

	def toStateRepresentation(self, rawState):
		return rawState # follow the representation of rawState.
		#raise NotImplementedError
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		if self.currentEpisode != episodeNumber:
			self.currentEpisode = episodeNumber
			self.numTakenActionsLastEpisode = numTakenActions
		numTakenActionsInepisode = numTakenActions - self.numTakenActionsLastEpisode
		epsilonComputed = np.power(0.9999,episodeNumber)
		#epsilonComputed = 0.2
		learningRateComputed = np.power(0.9999954,numTakenActions)
		return(learningRateComputed,epsilonComputed)
		#raise NotImplementedError

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	numAgents = args.numAgents
	numEpisodes = args.numEpisodes
	for i in range(numAgents):
		agent = JointQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0, numTeammates=args.numAgents-1)
		agents.append(agent)

	numEpisodes = numEpisodes
	numTakenActions = 0
	totalReward_record = []
	totalTimestep = []
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		totalReward = 0.0
		timeSteps = 0
			
		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())

			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			for agentIdx in range(args.numAgents):
				oppoActions = actions.copy()
				del oppoActions[agentIdx] # delete the current agent's action from the current actions
				agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions, 
					reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			totalReward = totalReward + reward[0]
			timeSteps = timeSteps + 1
			observation = nextObservation

		totalTimestep.append(timeSteps)
		totalReward_record.append(totalReward)
		if episode % 1000 == 0:
			print(episode)
			print(np.mean(totalReward_record))
			print(np.mean(totalTimestep))
			print(np.max(totalTimestep))
			totalReward_record = []
			totalTimestep = []
