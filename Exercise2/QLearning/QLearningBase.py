#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

class QLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(QLearningAgent, self).__init__()
		self.learningRatePassed = learningRate
		self.discountFactor = discountFactor
		self.epsilonPassed = epsilon
		self.S = [(x,y) for x in range(6) for y in range(5)] # 生成所有的state
		self.S.append("GOAL")
		self.S.append("OUT_OF_BOUNDS")
		self.curState = (0,0)
		self.experience = 0
		self.timeStep = 0
		self.qValue = {}
		self.curAction = 'None'
		for state in self.S:
			for action in self.possibleActions:
				self.qValue[(state,action)] = initVals

	def learn(self):
		t = self.timeStep - 1
		state_t = self.expeirence[t][0]
		action_t = self.expeirence[t][1]
		reward = self.experience[t][2]
		state_next = self.experience[t+1][0]

        q_value_next = []
		for action in self.possibleActions:
			q_value_next.append(self.qValue[(state_next,action)])
		q_value_nextMax = max(q_value_next)
		q_value_nextMaxIndexAll = [i for i,j in enumerate(q_value_next) if j==q_value_nextMax]
		q_value_nextMaxIndex = random.choice(q_value_nextMaxIndexAll)
        action_next = self.possibleActions[q_value_nextMaxIndex]
		
		qValue = self.qValue[(state_t,action_t)]
		changeValue =  self.learningRate*(reward+self.discountFactor*(self.qValue[(state_next,action_next)])-qValue)
		self.qValue[(state_t,action_t)] =qValue + changeValue
		return changeValue
		raise NotImplementedError

	def act(self):
		q_value = []
		for action in self.possibleActions:
			q_value.append(self.qValue[(self.curState,action)])
		q_valueMax = max(q_value)
		q_valueMaxIndexAll = [i for i,j in enumerate(q_value) if j==q_valueMax]
		q_valueMaxIndex = random.choice(q_valueMaxIndexAll)
		actionIndexAll = [i for i in range(self.possibleActions)]
		actionIndexAll.remove(q_valueMaxIndex)
		proNotMaxA = self.epsilon/self.amountActions
		proisMaxA = 1 - self.epsilon + proNotMaxA
		pro = random.random()
		if pro <= proisMaxA:
			self.curAction = self.possibleActions[q_valueMaxIndex]
		else:
			self.curAction = self.possibleActions[random.choice(actionIndexAll)]
		return self.curAction
		raise NotImplementedError

	def toStateRepresentation(self, state):
		return state
		raise NotImplementedError

	def setState(self, state):
		self.curState = state
		raise NotImplementedError

	def setExperience(self, state, action, reward, status, nextState):
		self.expeirence.append((self.timeStep,state,action,reward,nextState))
		self.timeStep = self.timeStep+1
		raise NotImplementedError

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
		raise NotImplementedError

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		raise NotImplementedError

	def reset(self):
		self.experience = []
		self.timeStep = 0
		raise NotImplementedError
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return(self.learningRatePassed,self.epsilonPassed)
		raise NotImplementedError

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes

	# Run training using Q-Learning
	numTakenActions = 0 
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()
		
		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			update = agent.learn()
			
			observation = nextObservation
	
