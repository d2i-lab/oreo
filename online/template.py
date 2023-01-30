import numpy as np


class OnlineTemplate:
    def __init__(self, tree_builder, states, alpha):
        self.tree_builder = tree_builder
        self.states = list(states)
        self.new_states = []
        self.to_delete = []
        # Unit movement cost
        self.alpha = alpha
        self.movement_cost = 0
        self.query_cost = 0
        # Index of current state
        self.idx = np.random.randint(len(states))
        self.curr_state = str(states[self.idx])
        self.schedule = {"query": [], "move": []}
        self.T = 0
        # Build initial layout
        self._build_layout()
        self.switch_layout()

    # Add a new state
    def add_state(self, state):
        self.new_states.append(state)

    # Delete a state
    def del_state(self, idx):
        # Deleting a state => setting counters to max
        self.counters[idx] = self.alpha
        self.to_delete.add(idx)
        if self.idx == idx:
            # Forced to move
            self.change_state()

    # Construct a new layout on the dataset. Incur alpha movement cost
    def _build_layout(self, sample=False):
        use_zorder = (self.tree_builder.args.policy != "oracle") and (\
                (self.tree_builder.args.method == "z") or (self.idx == 0))
        eval_tree = self.tree_builder.load_by_path(
            self.states[self.idx].path, use_zorder, sample)
        self.next = (eval_tree, self.states[self.idx].path)
        if self.T > 0:
            self.movement_cost += self.alpha

    def switch_layout(self):
        self.curr = self.next[0]
        self.schedule["move"].append((self.T, self.next[1]))

    def change_state(self):
        pass

    def process_queries(self, new_queries):
        pass

