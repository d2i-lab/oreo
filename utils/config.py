import json

class Args:
    """ Input arguments """
    def __init__(self, fname):
        with open("resources/params/%s.json" % fname, mode="r") as f:
            args = json.load(f)

        self.method = args["method"]
        self.config = args["config"]
        self.q = args["q"]
        self.k = args["k"]
        self.alpha = args["alpha"]
        self.eps = args["eps"]
        self.gamma = args["gamma"]
        self.policy = args["policy"]
        self.interval = args["interval"]
        self.res = args["res"]
        self.lag = args["lag"]
        self.seed = args["seed"]
        self.equal = (args["equal"] == 1)
        self.load = (args["load"] == 1)
        self.rewrite = False
        self.alg = "offline"
        self.n = 2000

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_eps(self, eps):
        self.eps = eps

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_interval(self, interval):
        self.interval = interval

    def set_res(self, res):
        self.res = res

    def set_policy(self, policy):
        if len(set(policy.split(",")).intersection(['sw', 'res', 'oracle'])) > 1:
            self.policy = policy
        else:
            raise ValueError("Unsupported policy: %s" % policy)

