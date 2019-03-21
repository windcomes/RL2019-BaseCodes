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
            #self.policy[state] = self.MDP.A

	def BellmanUpdate(self):
		for state in self.MDP.S:
			self.policy[state] = []
			all_values = []
			for a in self.MDP.A:
				s_r_sum = 0
				prob_next_states = self.MDP.probNextStates(state,a)
				for state_2 in prob_next_states.keys():
					s_r_sum = s_r_sum + prob_next_states[state_2] * (self.MDP.getRewards(state,a,state_2)+self.discountRate*self.values[state_2])
				all_values.append(s_r_sum)
				
			self.values[state] = max(all_values)
			for i in range(len(all_values)):
				if all_values[i] == self.values[state]:
					self.policy[state].append(self.MDP.A[i])

		return (self.values,self.policy)

		raise NotImplementedError
		
if __name__ == '__main__':
	solution = BellmanDPSolver(0.5)
	for i in range(20000):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values)
	print("Policy : ", policy)

