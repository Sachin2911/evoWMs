# Process for building the literature review

## The two things being produced

- **Annotated bibliography:** one short entry per paper, each on its own. These are working notes.
- **Literature review:** an argument across the papers, organised by theme, that ends at the gap this project fills. It later becomes the Related Work section of the GECCO paper and a chapter of the thesis.

The review has three jobs:

1. Find out whether the idea has already been done.
2. Find the benchmark and baselines reviewers will expect.
3. Find how results are expected to be measured and reported.

## The steps

| Step | What to do | Output |
|---|---|---|
| 1. Frame | Write the research question and the themes it touches | A list of 5–6 themes |
| 2. Collect | Start from seed papers. Follow their references backward and their "cited by" lists forward. Look for survey papers first. | 60–100 candidates in a reference manager (Zotero) |
| 3. Triage | 5–10 minutes per paper: title, abstract, introduction, headings, conclusion. Mark keep, maybe, or drop. | 25–35 papers kept |
| 4. Read | About an hour per kept paper. Several hours each for the 4–6 papers the project builds on or compares against. | Understanding |
| 5. Annotate | Write an entry straight after reading each paper | Annotated bibliography |
| 6. Synthesise | Spreadsheet with papers as rows and properties as columns | Synthesis matrix |
| 7. Write | Prose organised by theme, not paper by paper. Each section ends on what is unresolved. The whole review ends at the gap. | Literature review |

Stop collecting when new searches mostly return papers already on the list.

### Template for one annotation

About 150 words, in my own words:

- **Citation**
- **Question:** what the paper asks
- **Method:** two sentences
- **Evidence:** benchmark, baselines, metrics, headline numbers
- **Weaknesses**
- **Relation to my claim:** supports it, competes with it, is a baseline, or offers a method to borrow

### Columns for the synthesis matrix

- Frozen or retrained model
- Needs a hand-written cost function
- Has a penalty weight to tune
- Gradient-based or black-box
- Benchmark used
- Size of what is trained

The gap is the combination of cells that no paper fills. If a paper does fill it, that is learned in week 1 and not in January.