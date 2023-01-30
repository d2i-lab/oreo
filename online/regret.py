import numpy as np
from online.template import *
from offline.greedy import *


class Regret(OnlineTemplate):
    def __init__(self, tree_builder, states, alpha, fractor=1.):
        super().__init__(tree_builder, states, alpha)
        self.regret = [0] * len(states)
        self.query_history = []
        self.thresh = fractor * alpha
        self.use_sample = not self.tree_builder.args.load

    def change_state(self, new_idx):
        self.idx = new_idx
        self._build_layout(self.use_sample)
        self.switch_layout()
        self.regret = [0] * len(self.states)
        self.query_history = []

    def add_state(self, state):
        self.states.append(state)
        cum_cost = sum(state.eval(self.query_history, avg=False)[0])
        self.regret.append(cum_cost)

    # Process new queries and update counters
    def process_queries(self, new_queries):
        for q in new_queries:
            self.T += 1
            ref_cost, pids = self.curr.eval([q])
            self.query_cost += ref_cost
            self.schedule["query"].extend(pids)

            # Update all counters with query costs
            for j, tree in enumerate(self.states):
                read, _ = tree.eval([q])
                self.regret[j] += read

                # Check if need to change state
                if self.regret[self.idx] - self.regret[j] >= self.thresh:
                    self.change_state(j)
                    break

            self.query_history.append(q)

