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
- **Steering**: the hand-written adapter. Each action number is shifted towards the push that the look-ahead expects to cost less, by the difference between the two cost changes. It uses 4 of the 40 weights of the smallest adapter, and no search.

## Summary

| # | Question | Decision |
|---|---|---|
| 1 | What may the recipe assume about the frozen agent? | Five pieces every world-model agent has, plus a small cost predictor we train |
| 2 | What does the adapter read? | Look-ahead inputs as the main choice, the raw internal state as the comparison |
| 3 | How does the adapter act? | On the proposed action only. Residual against re-rank is a designed comparison. Re-rank chooses among pushed actions |
| 4 | How large is the adapter? | Three sizes: tens, hundreds and thousands of weights |
| 5 | How do safety levels come in? | A quality-diversity archive over cost levels, with a free dial as the baseline it must beat |
| 6 | How is hand-tuning avoided? | Remove knobs, then evolve them, then search them, on development tasks only |
| 7 | Where are candidates scored? | In the world model. Real episodes for the current best, the cost predictor and the final choice |
| 8 | Which tasks are used for what? | Develop on PointGoal1 and PointButton1, hold out everything else |
| 9 | What counts as a result? | Reward kept at SafeDreamer's cost limit, every real step counted, against SafeDreamer and against steering |
| 10 | Where does the search start? | From zero weights. Steering is a baseline, not a starting point |

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
- **Re-rank**: take the proposed action and the pushed versions of it that the look-ahead already imagines, score each, and take the best. The score is a weight times the change in predicted reward minus a weight times the change in predicted cost, both against the proposed action. The ratio of the two weights is the price of cost. The proposed action is kept unless another scores higher.
- **Mix**: a re-rank in which the residual's action is one more candidate.

**Why only on the proposed action.**

- **Fact:** the proposed action is the one thing both agent families produce in the same form. An adapter inside the network (low-rank weight changes, or shifting the state the actor reads) needs an actor network, and TD-MPC2's action comes out of a planner.
- The frozen agent stays a black box, so "nothing in the agent is changed" is literally true, and an all-zero adapter can be checked to replay the unadapted agent exactly.

**Why the form is a comparison and not a choice.**

- **Fact:** the two main forms have opposite strengths.
  - Residual can produce any action, so it can always steer away. But it can also produce actions the frozen model never saw, which is where the model's predictions are least reliable. It needs a bound on its output, which is one more value to set.
  - Re-rank picks among a fixed, small set of actions and needs no bound. Its price is a single dial for the safety level. But it can only take an action that is in the set.
- Re-rank has two weights, so it leaves little for evolution to find. Residual leaves more. This bears on what the paper can claim for evolution, and it makes re-rank with a swept price the cheapest adapter the evolved one has to beat.

**Why re-rank chooses among pushed actions, not among further actions from the agent.** The first design drew its candidates from the frozen agent, so as to stay where the model is reliable. Its assumption was that the agent's proposals are spread widely enough to contain a safe one.

- **Observation:** for DreamerV3 on both development tasks they are not. A further action from the agent lies about 0.11 from the proposed one in each action number, where a push moves a number by up to 2. At decisions where a cost is expected, 4 further actions offer a halving of the expected cost at 6% to 25% of them and 64 further actions at 15% to 48%. The four pushes offer it at 50% to 73%.
- **Observation:** in real episodes, taking the lowest-cost one of 4 further actions kept the reward, and lowered the cost clearly in only one of four runs.
- **Observation:** the worry that pushed actions leave the region where the model is reliable did not show. Imagined and real cost follow each other as closely for an action pushed to its limit and held as for the proposed one (correlation 0.91 to 0.94 on PointGoal1, 0.79 to 0.81 on PointButton1).
- **Fact:** the pushed actions are already imagined by the look-ahead, so re-ranking them costs no further roll-outs. Further actions from TD-MPC2 would each cost a run of its planner.

**Rests on.** The three observations, from code that has not been reviewed.

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

**What the first measurement showed.** 21 candidate adapters were scored in the model and played for 40 real episodes each.

- **Observation:** where candidates differ, the model orders them by cost about as well as the real episodes agree with themselves (rank correlation 0.82 to 0.88, against 0.89 between two halves of the real episodes). It picks out the safe candidates almost without error.
- **Observation:** it cannot order candidates whose real costs lie within a few units of each other. Its imagined cost stops falling at about 11 on PointGoal1, where the real cost is between 1 and 10. The cost limit of 2 lies in that range.
- **Observation:** its imagined reward is too high for adapters that change the agent's behaviour. Steering keeps 61% of the reward in real episodes on PointGoal1 and 86% in the model.
- So the model can bring a search to the safe adapters, and cannot choose among them or say that the limit is met. The real episodes of this decision are needed, not optional.

**To settle before the search.** The imagined roll-outs started from states of the unadapted agent, 6.6% of which were already being charged a cost. **Assumption:** that is where the floor under the imagined cost comes from, and roll-outs that start from the adapted agent's own real episodes do not have it. If they still have it, the search needs more real episodes than planned.

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

## 9. What counts as a result

**Decision.** The result that matters is the reward kept at SafeDreamer's cost limit of 2, with every real step counted. It is set against SafeDreamer and against steering. The aim is to reach it in under about 250,000 real steps in all, including the episodes that train the cost predictor.

**Why the cost limit.** **Fact:** SafeDreamer is trained for, and reports against, a cost limit of 2. Lower cost that stays above the limit does not make the agent safe by that measure, and reward is only comparable between agents at the same cost.

**Where the two development tasks stand.** Reward and cost of an episode. **Observation:** the first two rows are the released checkpoints on seeds 0 to 49, the third is from 40 episodes on other seeds.

| | PointGoal1 reward | PointGoal1 cost | PointButton1 reward | PointButton1 cost |
|---|---|---|---|---|
| Unadapted agent | 26.9 | 54.5 | 27.2 | 96.4 |
| SafeDreamer (BSRP-Lag) | 21.4 to 21.7 | 0.8 to 1.8 | 5.8 to 6.2 | 0.1 to 0.6 |
| Steering | 16.3 | 2.9 | 15.2 | 15.1 |

- On PointGoal1 steering is near SafeDreamer's cost with about three quarters of its reward. A result there means about 5 more points of reward at the same cost.
- On PointButton1 steering keeps 2.5 times SafeDreamer's reward at a cost of 15. A result there means bringing the cost to 2 while keeping more than 6 of reward.

**Why against steering.** Steering has no search in it. If the evolved adapter does not beat it, evolution is not the active ingredient, which is the weakness of the earlier draft paper (decision 2).

**Why 250,000 steps.** **Fact:** by the SafeDreamer paper's curves, SafeDreamer first gets under the limit on PointGoal1 after about 0.5M to 1M steps (decision 7), so this is at most half of that. **Observation:** a cost predictor trained on 20 episodes scores almost as well as one trained on 140 (AUROC 0.996 against 0.998 on PointGoal1, 0.971 against 0.987 on PointButton1), so its episodes need not take most of the budget.

## 10. Where the search starts

**Decision.** The search starts from zero weights, which is the unadapted agent. Steering is not its starting point. Steering and its dial, which scales its weights between 0 and 1, are baselines the search must beat.

**Why.**

- A search started from steering would carry a hand-chosen safe direction into the result. The claim that the adapter has no built-in safe direction would no longer hold, and a search that only improves a hand-written adapter cannot show that evolution found anything.
- **Observation:** the model orders candidates scattered around zero well by cost (rank correlation 0.86 to 0.93). Finding the safe adapters from zero is the part of the search the model can do without real episodes.
- **Observation:** steering is not the best setting of its own weights. On PointButton1, at half strength it costs about the same (17.1 against 15.1) and keeps more reward (19.4 against 15.2). There is something left for a search to find.

**Set aside.** *Starting from steering*: faster, but it gives up the claim above.

## Order of work

1. **Decisions 1, 7 and 8 first.** Decision 1 fixes what everything else may use. Decision 7 fixes how many candidates can be afforded, which bounds the sizes in decision 4 and decides whether the archive in decision 5 is affordable. Decision 8 fixes where tuning is allowed, and has to be in place before any more results are looked at.
2. **Decisions 2, 3 and 4 as one designed comparison** on the development tasks. They interact: re-rank has few weights whatever it reads, the raw state implies thousands of weights, and the look-ahead implies tens. Varying one at a time would mislead.
3. **Decision 5 after that.** An archive of adapters needs an adapter form that works at one safety level first.

Before any search, two things are measured with steering, because a search can only improve what the adapter's inputs and the scoring can show:

- What the cost that steering leaves on PointButton1 is made of (fixed hazards, moving obstacles, wrong buttons), and where its reward is lost on PointGoal1.
- Whether imagined roll-outs that start from the adapted agent's own states can tell low-cost adapters apart (decision 7).

## Baselines that test whether evolution matters

These are part of the design from the start, because they decide whether evolution is the active ingredient.

- **Random search**: the same adapter, scoring and budget, with candidates drawn at random. This shows whether CMA-ES adds anything beyond sampling.
- **Gradient training**: the same adapter trained by gradient descent through the world model. The adapter and the model are differentiable, so this is the obvious alternative.
- **The free dial** of decision 5, for the safety-level claim.
- **Steering and its dial**: the same inputs with hand-written weights and no search. This shows what the inputs give on their own (decision 10).
- **Re-rank with a swept price**: two weights and no search (decision 3).

## Assumptions to test

| Assumption | Decision it supports | What would test it |
|---|---|---|
| The frozen internal state holds enough to predict cost, in TD-MPC2 as well as DreamerV3 | 1 | The cost-predictor test repeated on TD-MPC2 |
| The world model ranks candidates in roughly the same order as the real environment | 7 | Rank agreement between imagined and real scores for the same candidates |
| Imagined roll-outs that start from the adapted agent's own states can tell low-cost adapters apart | 7 | The comparison of imagined and real scores again, with such starting states |
| The cost steering leaves on PointButton1 is of a kind the adapter's inputs can show | 2, 9 | That cost split by its source, in real episodes |
| An archive over cost levels keeps the search from collapsing to standing still | 5, 6 | The archive against a single-ranking search, same budget |
| Scaling a safe adapter's output moves cost steadily between the unadapted and adapted levels | 5 | Cost measured at several scales of one adapter |
| A small network can learn when to act, so no hand-built gate is needed | 4, 6 | The small network against the weighted sum with a gate |
