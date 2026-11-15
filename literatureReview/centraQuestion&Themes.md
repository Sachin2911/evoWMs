## Question
How much of a retrained world-model agent's safety can be recovered by evolving a small adapter on a frozen, reward-only one, and at what cost in reward and environment interaction?

## Hypothesis:
The evolved adapter matches the retrained agent's violation rate, losing no more than X% of its reward, using under Y% of the environment interaction.

## Question based Themes

1. **The specification problem:** What must be hand-specified to make an RL agent safe today?
2. **Safety by retraining:** How are world-model agents made safe at training time?
3. **Safety after training:** How is safety added to a fixed policy after training?
4. **Evolution under constraints:** What has evolution done under constraints, and in safe RL?
5. **Adapting frozen models:** How are frozen pretrained models adapted cheaply?
6. **Evaluation:** How are safe RL results expected to be measured and reported?

## Category Based Themes

1. **Safe RL basics and benchmarks:** penalty and constraint methods, Safety-Gymnasium
2. **World-model agents and their safe versions:** Dreamer, TD-MPC2, SafeDreamer
3. **Safety filters:** the latent safety filter line, reachability, barrier functions
4. **Evolution under constraints:** Constrained Cross-Entropy Method, Marchesini et al., AutoCost
5. **Adapting frozen models:** residual policies, adapters, evolution on adapters
6. **How results should be measured and reported**