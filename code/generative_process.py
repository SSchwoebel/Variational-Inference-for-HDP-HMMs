# this file contains the generative process code

import torch

class HMM:

    def __init__(self):

        self.states = []
        self.observations = []

    def set_parameters(self, pi, phi, prior_states):

        # transition matrix
        self.pi = pi

        # observation generation probabilities
        self.phi = phi

        # prior over states
        self.prior_states = prior_states

        # make sure the dimensions are consistent
        # for the state transition matrix
        assert self.pi.shape[0] == self.pi.shape[1], \
            "pi is not square: Dimension 0 is of size {0}, but dimension 1 is of size {1}"\
                .format(self.pi.shape[0], self.pi.shape[1])
        
        # for the observation probabilities
        assert self.pi.shape[0] == self.phi.shape[0], \
            "pi is inconsistent with phi: Dimension 0 of pi is {0}, but dimension 0 of phi is {1}"\
                .format(self.pi.shape[0], self.phi.shape[0])
        
        # for the prior over states
        assert self.pi.shape[1] == self.prior_states.shape[0], \
            "pi is inconsistent with phi: Dimension 1 of pi is {0}, but dimension 0 of prior_states is {1}"\
                .format(self.pi.shape[1], self.prior_states.shape[0])

    def initialize_process(self):
        # Note that this function generates the first state from the prior over states, and generates a first observation
        
        # initialize state list, keeps track of hidden state index
        start_state = torch.distributions.Categorical(probs=self.prior_states).sample()
        self.states = [start_state]

        # initialize observation list, keeps track of observation indices
        self.observations = []
        # generate first observation
        self.generate_observation()
        

    def transition_state(self):

        # generate new state from transition probabilities
        new_state = torch.distributions.Categorical(probs=self.pi[:,self.states[-1]]).sample()

        # save into list
        self.states.append(new_state)

        # check that there there is one less observations than there are states
        assert len(self.states) == len(self.observations)+1,\
            "the ammount of observations generated does ({0}) not match the amount of states ({1}) (should be one more state than observation)"\
                .format(len(self.observations), len(self.states))
        

    def generate_observation(self):

        # generate observation from state
        curr_obs = torch.distributions.Categorical(probs=self.phi[:,self.states[-1]]).sample()

        # save into list
        self.observations.append(curr_obs)

        # check that there are as many observations as there are states
        assert len(self.states) == len(self.observations),\
            "the ammount of observations generated does ({0}) not match the amount of states ({1})"\
                .format(len(self.observations), len(self.states))
        
        
    def evolve(self):

        self.transition_state()
        self.generate_observation()


    def simulate_timeseries(self, T):

        if len(self.states)>1 or len(self.observations)>1: 
            print("existing time series will be deleted")

        self.initialize_process()
        
        for t in range(T-1):
            self.evolve()
