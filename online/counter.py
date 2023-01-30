import numpy as np
from online.template import *
from offline.greedy import *


class CounterManager(OnlineTemplate):
    """RANDOM with state modification"""
    def  __init__(self, tree_builder, states, alpha, temperature=1, lag=0, smoothing=1e-2):
        super().__init__(tree_builder, states, alpha)
        self.counters = [0] * len(states)
        self.cost_est = [0] * len(states)
        self.new_counters = []
        self.smoothing = smoothing
        self.temperature = temperature
        self.lag = lag
        self.switch_countdown = 0
        self.use_sample = not self.tree_builder.args.load
        
    # New phase: include states that were added during the last phase
    def _start_new_phase(self):
        new_states = []
        new_cost_est = []
        # Current state is deleted - have to move in the new phase
        if self.idx in self.to_delete:
            self.curr_state = None
        # Delete states from the last phase
        for idx, s in enumerate(self.states):
            if not idx in self.to_delete:
                new_states.append(s)
                new_cost_est.append(self.counters[idx])
                # Update index pointer in the new state space
                if self.curr_state == str(s):
                    self.idx = len(new_states) - 1
        # Add new states
        new_states.extend(self.new_states)
        new_cost_est.extend(self.new_counters)
        self.states = new_states
        self.cost_est = new_cost_est
        # Reset counters
        self.counters = [0] * len(self.states)
        self.new_states = []
        self.new_counters = []
        self.to_delete = []

    # End phase when all counters reach unit movement cost
    def _end_phase(self):
        return np.min(self.counter) >= self.alpha

    def add_state(self, state):
        self.new_states.append(state)
        self.new_counters.append(np.median(self.counters))
        
    # When counter is full, switch to a non-full counter with uniform probability
    def change_state(self):
        indices = np.where(np.array(self.counters) < self.alpha)[0]
        if len(indices) == 0:
            self._start_new_phase()
            # Heuristic: stay in current state in the new phase if current state performs well
            if self.cost_est[self.idx] > np.average(self.cost_est) + self.smoothing:
                indices = list(range(len(self.states)))
            else:
                indices = [self.idx]
        else:
            self.cost_est = self.counters

        cost_est = (np.array(self.cost_est)[indices] + self.smoothing).flatten()
        weights = (np.max(cost_est) - cost_est) / (np.sum(np.max(cost_est) - cost_est) + self.smoothing)
        weights = weights**self.temperature / np.sum(weights**self.temperature)
        if np.any(weights != weights):
            weights = np.ones(cost_est.shape) * 1. / cost_est.shape[0]

        new_idx = np.random.choice(indices, 1, p=weights)[0]
        # Check whether we have moved to a different state
        if self.curr_state != str(self.states[new_idx]):
            # print("[T=%d] Decide to change state: %d=>%d" % (self.T, self.idx, new_idx))
            self.idx = new_idx
            self.curr_state = str(self.states[new_idx])
            self._build_layout(self.use_sample)
            self.switch_countdown = self.lag

    # Process new queries and update counters
    def process_queries(self, new_queries):
        for q in new_queries:
            self.switch_countdown -= 1
            # Update all counters with query costs
            for j, tree in enumerate(self.states):
                read, _ = tree.eval([q])
                self.counters[j] += read
                
            for j, tree in enumerate(self.new_states):
                read, _ = tree.eval([q])
                self.new_counters[j] += read

            # Check if need to change state
            if self.counters[self.idx] >= self.alpha:
                self.change_state()
            if self.switch_countdown == 0:
                # print("[T=%d] switching state" % (self.T) )
                self.switch_layout()

            read, pids = self.curr.eval([q])
            self.query_cost += read
            self.schedule["query"].extend(pids)
            self.T += 1
