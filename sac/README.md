Soft Actor Critic uses the maximum entropy framework
- Where entropy means disorder
- Entropy encourages exploration

Reward curves of SAC agents during training tend to have much less rolling variance

The goal of SAC agents is to not just to maximise the total reward but the stochasticity of the agent's behaviour over time

Instead of outputting the actions directly, the neural networks in SAC agents output a mean and standard deviation for a normal distribution that is then sampled to get actions.

The critic network takes a state and action as an input
- And outputs a value indicating whether the action taken was good or terrible

The value network takes a state as an input
- And outputs a value indicating whether the state is valuable or not

The interplay of the three networks produces an agent that predicts the best sequence of actions to take over time.

## Questions
Does the process of sampling actions from a probability distribution over actions categorise an MDP as a stochastic MDP? 
- Stochastic policy?