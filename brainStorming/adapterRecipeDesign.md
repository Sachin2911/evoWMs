# Adapter recipe: design decisions

The recipe is the method for making a frozen, reward-only world-model agent safe with a small evolved adapter. This file records the design decisions for it and the reasoning behind each one.

The reasoning uses three kinds of statement, labelled where it matters:

- **Fact**: follows from an agent's architecture, from a paper, or from arithmetic.
- **Observation**: seen in our own runs. The code behind those runs has not been reviewed, so no decision rests on an observation alone.
- **Assumption**: believed but untested. The assumptions are collected at the end.

## Terms

- **Frozen agent**: a trained agent whose weights are never changed. Here it is a DreamerV3 agent trained on reward only, and later a TD-MPC2 agent.
- **Unadapted agent**: the frozen agent acting with no adapter.
- **Adapter**: a small set of weights that changes the action the frozen agent proposes. With every weight at zero it changes nothing.
- **Internal state**: the world model's compact summary of what it has seen. The papers call it the latent state.
- **Cost and cost limit**: cost counts the unsafe steps in an episode. The cost limit is the highest average episode cost that counts as safe (2 in SafeDreamer). A safety level is a choice of cost limit.
- **Step, episode, budget**: a step is one simulator step and an episode is 1,000 of them. The budget is the number of real steps a method uses.
- **CMA-ES**: the evolutionary search used here. It samples candidate weight settings around a centre, ranks them, and moves the centre and the shape of its sampling towards the better ones.
- **Quality diversity**: an evolutionary search that fills an archive of cells. Each cell keeps the best candidate found for one kind of behaviour, so the result is many good candidates and not one.

## Summary

| # | Question | Decision |
|---|---|---|
| 1 | What may the recipe assume about the frozen agent? | Five pieces every world-model agent has, plus a small cost predictor we train |
| 2 | What does the adapter read? | Look-ahead inputs as the main choice, the raw internal state as the comparison |
| 3 | How does the adapter act? | On the proposed action only. Residual against re-rank is a designed comparison |
| 4 | How large is the adapter? | Three sizes: tens, hundreds and thousands of weights |
| 5 | How do safety levels come in? | A quality-diversity archive over cost levels, with a free dial as the baseline it must beat |
| 6 | How is hand-tuning avoided? | Remove knobs, then evolve them, then search them, on development tasks only |
| 7 | Where are candidates scored? | In the world model. Real episodes for the current best, the cost predictor and the final choice |
| 8 | Which tasks are used for what? | Develop on PointGoal1 and PointButton1, hold out everything else |

## 1. What the recipe may assume about the frozen agent

**Decision.** The recipe uses only five pieces of the frozen agent: its internal state, its model that rolls that state forward given actions, its reward predictor, its value estimate, and the action it proposes. To these we add one small cost predictor trained on the internal state. Nothing specific to DreamerV3 is used.

**Why.**

- **Fact:** DreamerV3 and TD-MPC2 both have these five pieces. They differ in what proposes the action: an actor network in DreamerV3, a planner in TD-MPC2.
- **Fact:** neither agent has anything for cost, so some cost signal has to be added. A small predictor reading the frozen internal state is the smallest addition that provides one.
- The paper's second part applies the same recipe to TD-MPC2. If the recipe read pieces only DreamerV3 has (the split of its state into continuous and categorical parts, its image decoder, the meaning of the Point robot's two action numbers), the second part would test a different method from the first.
- The claim "this works on frozen world-model agents" is only as wide as what the recipe assumes about them.

**Rests on.** The assumption that the frozen internal state holds enough to predict cost. **Observation:** our cost-predictor test supports this for DreamerV3 on the five core tasks. It is untested for TD-MPC2, whose state is not shaped by image reconstruction.

## 2. What the adapter reads

**Decision.** The main input is the look-ahead: at each decision the frozen world model is rolled forward a few decisions with the proposed action, and again with each action number pushed up and down, and the adapter reads how the predicted cost and reward change. The raw internal state is kept as a second input type, for comparison.

**Why look-ahead is the main input.**

- **Fact:** a "danger is near" signal is one number about the state. It is the same whether the hazard is to the left or the right, so an adapter reading only that can change how strongly the agent acts but not which way. Avoiding a hazard without stopping needs direction.
- **Fact:** direction is available in two places. It is implicit in the raw internal state, where the adapter must learn to decode it from hundreds or thousands of numbers. It is explicit in the look-ahead, one pair of numbers per action number.
- **Fact:** the look-ahead grows with the number of action numbers, not with the size of the state. It needs 2A + 1 short roll-outs for A action numbers (5 for Safety-Gymnasium, 9 for Meta-World) and gives the adapter roughly 4A + 1 inputs. That keeps the adapter at about 50 weights for two action numbers and about 150 for four, which is where CMA-ES is strongest (decision 4).
- **Fact:** it uses only pieces from decision 1, so it carries over to TD-MPC2 unchanged.
- **Fact:** where changing the action does not change the predicted cost, the difference inputs are zero. The adapter is then quiet away from danger by construction, without a hand-built switch.

**Why the raw state is kept as a comparison.**

- **Fact:** the look-ahead is in effect a numerical gradient of predicted cost with respect to the action. An adapter on top of it only has to learn how far to move along that gradient.
- That makes the recipe open to the objection that the input design does the work and evolution adds little. **Observation:** the earlier draft paper had this weakness.
- An adapter on the raw state gets no such help. Running both shows how much of the result comes from the inputs and how much from the search.

**Set aside.**

- *Danger signals as the only input*: no direction, as above.
- *Cost so far in the episode*: only useful when the limit leaves something to spend. At a limit of 2 in 1,000 steps there is almost nothing. Worth revisiting for looser safety levels.
- *The requested safety level as an input*: belongs to the single level-conditioned adapter, set aside in decision 5.

**Cost.** The look-ahead adds roll-outs at every decision, so episodes run slower. **Observation:** about twice as slow.

## 3. How the adapter acts

**Decision.** The adapter acts only on the action the frozen agent proposes, never inside the agent. Which form it takes is settled by a designed comparison of three forms: residual, re-rank and a mix.

- **Residual**: add to, or scale, the proposed action.
- **Re-rank**: draw several candidate actions from the frozen agent, score each with predicted reward minus a price times predicted cost, and take the best.
- **Mix**: re-rank over the agent's candidates plus residual-shifted ones.

**Why only on the proposed action.**

- **Fact:** the proposed action is the one thing both agent families produce in the same form. An adapter inside the network (low-rank weight changes, or shifting the state the actor reads) needs an actor network, and TD-MPC2's action comes out of a planner.
- The frozen agent stays a black box, so "nothing in the agent is changed" is literally true, and an all-zero adapter can be checked to replay the unadapted agent exactly.

**Why the form is a comparison and not a choice.**

- **Fact:** the two main forms have opposite strengths.
  - Residual can produce any action, so it can always steer away. But it can also produce actions the frozen model never saw, which is where the model's predictions are least reliable. It needs a bound on its output, which is one more value to set.
  - Re-rank only picks among actions the agent itself proposes, so it stays where the model is reliable and needs no bound. Its price is a single dial for the safety level. But it has only as much authority as the spread of the agent's proposals: if every candidate heads into the hazard, there is nothing safe to pick.
- **Fact:** how wide that spread is depends on the agent. DreamerV3's actor has a learned spread, and TD-MPC2's planner keeps a population of candidates. So the answer may differ between the two families, and argument cannot settle it.
- Re-rank has very few weights, so it leaves little for evolution to find. Residual leaves more. This bears on what the paper can claim for evolution.

**The comparison.** Same inputs, same budget, same tasks and same search. Only the form differs.

## 4. How large the adapter is

**Decision.** Size is an experimental factor with three levels:

- **Tens of weights**: a weighted sum of the look-ahead inputs.
- **Hundreds**: a small network with one hidden layer on the same inputs.
- **Thousands**: a low-rank map from the raw internal state.

**Why.**

- **Fact:** for the weighted-sum form, the number of weights is (inputs + 1) × 2 × action numbers.
- **Fact:** CMA-ES learns how weights interact using a table that grows with the square of their number. That is practical up to the low thousands. Beyond that only a simplified version is usable, which ignores the interactions. The number of candidates needed also grows with the number of weights.
- So size trades what the adapter can express against what the search can afford. Argument cannot place the best point, which is why it is a factor.
- *Tens* is the smallest adapter that can express "steer by the look-ahead".
- *Hundreds* adds conditional behaviour, such as acting only when danger is high. That replaces the hand-built gate, which is a hand-set choice of signal and threshold (decision 6).
- *Thousands* is the raw-state comparison from decision 2. It is low-rank because a full map from DreamerV3's 2,816 state numbers would need over 11,000 weights for two action numbers, beyond what CMA-ES handles well.
- **Fact:** there is precedent for this range. Ha and Schmidhuber (2018) trained an 867-weight controller with CMA-ES on a frozen world model.
- How few weights are enough is a result worth reporting, since SafeDreamer retrains the whole agent.

## 5. How safety levels come in

**Decision.** The headline method is a quality-diversity search whose archive has one cell per cost level, each cell holding the highest-reward adapter found at that level. Its baseline is a free dial: take one safe adapter and scale its output between 0 and 1.

**Why a population method.**

- **Fact:** a safety level is a cost limit, and SafeDreamer needs a separate training run for each limit. Its cost-limit ablations (Figures 21 and 22) are separate runs.
- Evolution already keeps a population, so returning many adapters is natural for it and not for gradient training. This is where the evolutionary approach has a claim of its own.

**Why quality diversity and not a plain two-objective search.**

- **Fact:** an archive with a cell per cost level covers every level by construction. A two-objective front can crowd where the trade-off is easy and leave gaps elsewhere.
- **Fact:** a cell only needs "higher reward wins". There is no rule for trading reward against cost, and so none of the constants such a rule needs (decision 6).
- **Assumption:** cells at higher cost levels hold adapters that still move, and these act as stepping stones that keep the search from collapsing to standing still. **Observation:** our first pilot collapsed that way with a single ranking.
- **Fact:** there is precedent for quality diversity with a learned model. Smith et al. (2023) use a learned dynamics model to choose which candidates to try on a real robot.

**Why the free dial is the baseline.**

- **Fact:** an all-zero adapter is the unadapted agent. Scaling a safe adapter's output from 0 to 1 therefore traces a path of safety levels with no extra search.
- If that path is as good as the archive, the archive is not needed. A headline claim should be tested against its cheapest alternative.

**Set aside.**

- *One search per limit*: this is SafeDreamer's cost structure, so it offers no advantage. It stays only as a reference point.
- *One adapter that reads the requested level*: a harder search, because the adapter must be good at every level at once. It can be added later if the archive works.

## 6. How hand-tuning is avoided

**Decision.** Hand-set values are dealt with in a fixed order of preference: remove them, then evolve them, then search them. Any search over them happens on the development tasks only (decision 8), after which the recipe is frozen.

**Why this matters.**

- **Observation:** the exploratory recipe has about a dozen hand-set values (the gate and its threshold, the look-ahead length and push size, the output bound, input scales, the steps of the ranking's price, the search's step size and population). They were chosen while watching results on PointGoal1.
- A recipe tuned on a task and reported on the same task overstates what it does.
- A recipe that needs re-tuning for each task undercuts the claim of being cheaper than retraining, because tuning runs use real steps too.

**Why this order.**

- *Remove.* A value that is gone cannot be set wrongly and costs nothing.
  - Measure reward and cost as shares of the unadapted agent's, so the same number means the same thing on every task. **Observation:** across the five core tasks the unadapted agent's reward runs from about 23 to 35 and its cost from about 40 to 190, so fixed scales would not carry over.
  - Scale the adapter's inputs from the unadapted agent's recorded episodes, not by fixed constants.
  - The archive of decision 5 removes the ranking rule.
- *Evolve.* A value placed among the weights is tuned for each task inside the budget that is already counted. Candidates are the look-ahead length and any threshold. The search's step size already adapts itself in CMA-ES.
- *Search.* An outer search costs extra runs, so it is kept for what is left, such as population size and the number of imagined starting states.

**Why development tasks only.** The recipe is frozen before it meets a held-out task. That is what makes "no hand-tuning" something a reader can check.

## 7. Where candidates are scored

**Decision.** Candidates are compared inside the world model. Real episodes are used for three things only: the current best candidate, retraining the cost predictor, and the final choice. Every real step is counted in the budget.

**Why not score every candidate in the real environment.**

- **Fact:** an episode is 1,000 steps. By the SafeDreamer paper's curves, SafeDreamer gets under the limit on PointGoal1 in about 0.5M to 1M steps. A budget that clearly beats that, say 200,000 steps, is 200 episodes in total.
- **Fact:** twelve candidates on 2 episodes each is 24 episodes a generation, so that budget allows 8 generations. That is too few for any search, whatever the adapter.
- **Observation:** episode cost varies a great deal (for the unadapted agent on PointGoal1, a standard deviation of about 44 around a mean of about 55), so 2 episodes cannot rank candidates by cost. More episodes each would leave even fewer generations.

**Why not score everything in the world model.**

- **Fact:** scoring in the model uses no real steps, and every candidate can be given the same starting states and the same random numbers. Differences between candidates then come from the candidates, which is what a search by ranking needs.
- **Fact:** the model has no cost output, so the cost predictor must be trained on real episodes in any case.
- **Fact:** the model learned from the reward-only agent's experience, so it is reliable near what that agent does. The adapter's job is to change that behaviour, which moves the agent away from there.
- **Fact:** a search finds whatever scores well, including candidates that look safe only because the model is wrong about them. The model cannot be the final judge.

**Why this mix.**

- The many comparisons between candidates go to the model, where they are free and fair.
- The current best plays real episodes. This shows whether imagined progress is real, and it gives new data from where the adapted agent goes, which retrains the cost predictor and counters the drift.
- The final choice among a few finalists is made on fresh real episodes. Choosing on the model, or on episodes already used to steer the search, would be optimistic.

**Open problem.** An archive costs more to check than one adapter: 10 adapters on 20 episodes each is already 200,000 steps. Possible answers are to check only the levels asked for, or to share the same real episodes between neighbouring cells. Not decided.

**Rests on.** The assumption that the model ranks candidates in roughly the same order as the real environment, near states the agent has visited. This has to be measured and reported.

## 8. Which tasks are used for what

**Decision.** PointGoal1 and PointButton1 are the development tasks: every design choice and all tuning happen there. Every other task is held out. The 13 further Safety-Gymnasium tasks are not looked at until the recipe is frozen.

**The tasks.**

- **Fact:** five core tasks have reward-only and SafeDreamer checkpoints and are the SafeDreamer paper's main benchmark: PointGoal1, PointGoal2, PointButton1, PointPush1 and CarGoal1.
- **Fact:** 13 further Safety-Gymnasium tasks have reward-only and BSRP-Lag checkpoints. They add a third robot (Racecar) and the harder level-2 layouts.
- **Fact:** Meta-World through TD-MPC2 adds a second model family, four action numbers, a planner in place of an actor, and a cost we define ourselves.

**Why two development tasks.** One task cannot tell a general recipe from one fitted to that task.

**Why these two.**

- They bracket the range of difficulty. PointGoal1 has static hazards and is the simplest. PointButton1 has moving obstacles and wrong buttons, so its costs do not come from a fixed place.
- **Fact:** PointButton1 is also the hardest of the Point tasks for SafeDreamer itself. In the paper's Figure 18, BSRP-Lag keeps less than half of DreamerV3's reward there.
- A recipe tuned only on the easy task could build in "hazards are fixed circles" without anyone noticing.
- Both use the Point robot. A new robot, new layouts, a new kind of task and a new model family are therefore all left to the held-out tasks, which makes those a real test.

**Why the 13 stay unseen.** They are the only tasks that no design choice has touched. Once looked at, they cannot be unseen. They also have matched checkpoints, so the comparison with BSRP-Lag is available there.

**Limits.**

- PointPush1, PointGoal2 and CarGoal1 have already been used in exploratory runs. They are held out from tuning from here on, but they are not untouched. The 13 further tasks and Meta-World are.
- There is one reward-only checkpoint per task, so the effect of the task cannot be separated from the effect of that one training run.

## Order of work

1. **Decisions 1, 7 and 8 first.** Decision 1 fixes what everything else may use. Decision 7 fixes how many candidates can be afforded, which bounds the sizes in decision 4 and decides whether the archive in decision 5 is affordable. Decision 8 fixes where tuning is allowed, and has to be in place before any more results are looked at.
2. **Decisions 2, 3 and 4 as one designed comparison** on the development tasks. They interact: re-rank has few weights whatever it reads, the raw state implies thousands of weights, and the look-ahead implies tens. Varying one at a time would mislead.
3. **Decision 5 after that.** An archive of adapters needs an adapter form that works at one safety level first.

## Baselines that test whether evolution matters

These are part of the design from the start, because they decide whether evolution is the active ingredient.

- **Random search**: the same adapter, scoring and budget, with candidates drawn at random. This shows whether CMA-ES adds anything beyond sampling.
- **Gradient training**: the same adapter trained by gradient descent through the world model. The adapter and the model are differentiable, so this is the obvious alternative.
- **The free dial** of decision 5, for the safety-level claim.

## Assumptions to test

| Assumption | Decision it supports | What would test it |
|---|---|---|
| The frozen internal state holds enough to predict cost, in TD-MPC2 as well as DreamerV3 | 1 | The cost-predictor test repeated on TD-MPC2 |
| The world model ranks candidates in roughly the same order as the real environment | 7 | Rank agreement between imagined and real scores for the same candidates |
| The frozen agent's proposals are spread widely enough for re-rank to find a safe one | 3 | The share of decisions near a hazard at which any candidate lowers the predicted cost |
| An archive over cost levels keeps the search from collapsing to standing still | 5, 6 | The archive against a single-ranking search, same budget |
| Scaling a safe adapter's output moves cost steadily between the unadapted and adapted levels | 5 | Cost measured at several scales of one adapter |
| A small network can learn when to act, so no hand-built gate is needed | 4, 6 | The small network against the weighted sum with a gate |
