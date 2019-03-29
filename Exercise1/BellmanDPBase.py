from MDP import MDP

class BellmanDPSolver(object):
	def __init__(self, discountRate):
		self.MDP = MDP()
		self.discountRate = discountRate
		self.initVs()

	def initVs(self):
		self.values = {}
		self.policy = {}
		for state in self.MDP.S:
			self.values[state] = 0

	def BellmanUpdate(self):
		for state in self.MDP.S:
			self.policy[state] = []
			values_all = []
			for action in self.MDP.A:
				s_r_sum = 0
				prob_next_states = self.MDP.probNextStates(state,action)
				for state_2 in prob_next_states.keys():
					s_r_sum = s_r_sum + prob_next_states[state_2] * (self.MDP.getRewards(state,action,state_2)+self.discountRate*self.values[state_2])
				values_all.append(s_r_sum)
				
			self.values[state] = max(values_all)
			for i in range(len(values_all)):
				if values_all[i] == self.values[state]:
					self.policy[state].append(self.MDP.A[i])

		return (self.values,self.policy)

		raise NotImplementedError
		
if __name__ == '__main__':
	solution = BellmanDPSolver(0.5)
	for i in range(20000):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values)
	print("Policy : ", policy)

