"""Turns the recordings made by evaluate.py --record into the files of the Hugging Face dataset.

  python3 build_dataset.py --recordings FOLDER --earlier FOLDER... --out FOLDER [--upload ACCOUNT/NAME]

In --recordings, <checkpoint name>/ is what evaluate.py wrote for one checkpoint. --earlier
names folders holding the results files of an earlier, separate run of the same episodes;
each episode is marked with whether the two runs gave the same result. The output is:

  checkpoints.csv        one row per checkpoint, with its average reward and cost
  episodes.csv           one row per episode
  layouts.csv            one row per object per world: where everything started
  steps/<task>.parquet   one row per simulator step
  videos/                the videos, and metadata.csv describing them
  README.md              the dataset's page, from dataset_card.md

It stops if anything is missing or inconsistent. Run by build_dataset.sh, which supplies the
two packages this needs and the image lacks (pyarrow, huggingface_hub).
"""

import argparse
import csv
import json
import pathlib
import shutil

import numpy as np

from recording import make_folder

HERE = pathlib.Path(__file__).parent

# The four kinds of agent, in the order the tables list them, with the names used on the page.
VARIANTS = {
    'dreamerv3': 'DreamerV3 (reward only)',
    'bsrp_lag': 'SafeDreamer BSRP-Lag',
    'osrp_lag': 'SafeDreamer OSRP-Lag',
    'osrp': 'SafeDreamer OSRP',
}

# Every kind of cost the five tasks can charge. A step table has a column for each, so that
# the tables of all tasks have the same columns; a kind a task does not have stays 0.
COST_KINDS = ['cost_hazards', 'cost_vases_contact', 'cost_vases_velocity', 'cost_gremlins', 'cost_buttons']


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_csv(path, rows):
    make_folder(path.parent)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def earlier_results(folders):
    """The reward and cost of every episode in an earlier run, by checkpoint and episode seed."""
    results = {}
    for folder in folders:
        for path in sorted(folder.rglob('episodes.jsonl')):
            for episode in read_lines(path):
                results[episode['checkpoint'], episode['seed']] = (episode['reward'], episode['cost'])
    return results


def load_run(folder, episodes_wanted, checksums, earlier):
    """Everything recorded for one checkpoint, checked for completeness."""
    run = json.loads((folder / 'run.json').read_text())
    # 'safetygym_SafetyPointGoal1-v0' -> 'PointGoal1'
    run['task'] = run['task'].split('Safety', 1)[1].split('-')[0]
    matches = [(path, checksum) for path, checksum in checksums.items() if path.endswith('/' + run['checkpoint'])]
    assert len(matches) == 1, f'{run["checkpoint"]} is not in the checksum list exactly once'
    run['file'], run['sha256'] = matches[0]

    run['episodes'] = sorted(read_lines(folder / 'episodes.jsonl'), key=lambda episode: episode['seed'])
    seeds = [episode['seed'] for episode in run['episodes']]
    assert seeds == list(range(episodes_wanted)), f'{folder.name}: episode seeds are {seeds}'
    for episode in run['episodes']:
        key = (run['checkpoint'], episode['seed'])
        assert key in earlier, f'{folder.name} seed {episode["seed"]} is not in the earlier run'
        episode['replayed_exactly'] = earlier[key] == (episode['reward'], episode['cost'])
    # A restarted job can write an episode's layout twice; keep one per seed.
    run['layouts'] = {layout['seed']: layout for layout in read_lines(folder / 'layouts.jsonl')}
    assert sorted(run['layouts']) == seeds, f'{folder.name}: layouts do not match the episodes'
    run['folder'] = folder
    return run


def check_steps(run, episode, steps):
    """The recorded steps of an episode must add up to the episode's result."""
    name = f'{run["folder"].name} seed {episode["seed"]}'
    assert steps['cost'].sum() == episode['cost'], f'{name}: costs do not add up'
    assert abs(steps['reward'].astype(np.float64).sum() - episode['reward']) < 1e-2, f'{name}: rewards do not add up'
    kinds = [key for key in steps.files if key.startswith('cost_')]
    assert set(kinds) <= set(COST_KINDS), f'{name}: unknown kind of cost in {kinds}'
    # The simulator charges 1 on a step if anything caused a cost, and marks each cause with a 1.
    caused = sum(steps[key] for key in kinds)
    assert np.array_equal(caused > 0, steps['cost'] > 0), f'{name}: the causes of cost do not match the cost'


def steps_table(runs):
    """The step-by-step table for one task, all its checkpoints together."""
    import pyarrow as pa
    loaded = []
    for run in runs:
        for episode in run['episodes']:
            steps = np.load(run['folder'] / 'steps' / f'seed_{episode["seed"]:04d}.npz')
            check_steps(run, episode, steps)
            loaded.append((run, episode, steps))

    columns = {}

    def add(name, values, kind):
        columns.setdefault(name, []).append(pa.array(values, type=kind))

    for run, episode, steps in loaded:
        count = len(steps['reward'])
        add('task', [run['task']] * count, pa.string())
        add('variant', [run['variant']] * count, pa.string())
        add('training_seed', [run['training_seed']] * count, pa.int32())
        add('episode_seed', [episode['seed']] * count, pa.int32())
        add('step', np.arange(1, count + 1), pa.int32())
        add('decision', steps['decision'], pa.int32())
        for index in range(steps['action'].shape[1]):
            add(f'action_{index}', steps['action'][:, index], pa.float32())
        add('reward', steps['reward'], pa.float32())
        add('cost', steps['cost'], pa.float32())
        for name in COST_KINDS:
            add(name, steps[name] if name in steps.files else np.zeros(count), pa.float32())
        for index, name in enumerate(['agent_x', 'agent_y', 'agent_heading', 'agent_vx', 'agent_vy']):
            add(name, steps['agent'][:, index], pa.float32())
        add('goal_x', steps['goal'][:, 0], pa.float32())
        add('goal_y', steps['goal'][:, 1], pa.float32())
        add('goal_met', steps['goal_met'], pa.bool_())
        # x, y of each object that can move, in the order layouts.csv lists the movable objects.
        add('movable_xy', list(steps['movable']), pa.list_(pa.float32()))
    return pa.table({name: pa.chunked_array(chunks) for name, chunks in columns.items()})


def layout_rows(runs):
    """Where everything started, per task and episode seed. The same for every checkpoint."""
    rows = []
    for seed in sorted(runs[0]['layouts']):
        layout = runs[0]['layouts'][seed]
        for run in runs[1:]:
            assert run['layouts'][seed] == layout, (
                f'{run["task"]} seed {seed}: {run["folder"].name} started in a different world')
        x, y, heading = layout['agent']
        rows.append({'task': runs[0]['task'], 'episode_seed': seed, 'object': 'agent', 'index': 0,
                     'movable': True, 'size': '', 'x': x, 'y': y, 'heading': heading})
        for item in layout['objects']:
            rows.append({'task': runs[0]['task'], 'episode_seed': seed, **item, 'heading': ''})
    return rows


def results_table(checkpoint_rows):
    """The table of averages shown on the dataset's page."""
    lines = ['| Task | Agent | Training seed | Trained for (steps) | Reward | Cost |',
             '|---|---|---|---|---|---|']
    for row in checkpoint_rows:
        lines.append(
            f'| {row["task"]} | {VARIANTS[row["variant"]]} | {row["training_seed"]} '
            f'| {row["training_simulator_steps"] / 1e6:.1f}M '
            f'| {row["reward_mean"]:.2f} ± {row["reward_sd"]:.2f} '
            f'| {row["cost_mean"]:.2f} ± {row["cost_sd"]:.2f} |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--recordings', required=True, type=pathlib.Path)
    parser.add_argument('--earlier', required=True, type=pathlib.Path, nargs='+',
                        help='folders with the results files of an earlier run of the same episodes')
    parser.add_argument('--out', required=True, type=pathlib.Path)
    parser.add_argument('--episodes', type=int, default=50, help='episodes every checkpoint must have')
    parser.add_argument('--upload', help='Hugging Face dataset to upload to, as ACCOUNT/NAME; created private')
    args = parser.parse_args()

    checksums = {}
    for line in (HERE / 'checkpoints.sha256').read_text().splitlines():
        checksum, path = line.split()
        checksums[path] = checksum

    earlier = earlier_results(args.earlier)
    runs = [load_run(folder, args.episodes, checksums, earlier)
            for folder in sorted(args.recordings.iterdir()) if (folder / 'run.json').exists()]
    assert len(runs) == len(checksums), f'{len(runs)} recordings for {len(checksums)} checkpoints'
    # Ordered as in the checksum list: by task, then kind of agent, then training seed.
    order = list(checksums)
    runs.sort(key=lambda run: order.index(run['file']))
    tasks = list(dict.fromkeys(run['task'] for run in runs))

    if args.out.exists():
        shutil.rmtree(args.out)
    make_folder(args.out)

    checkpoint_rows, episode_rows, video_rows = [], [], []
    for run in runs:
        same = {key: run[key] for key in ('task', 'variant', 'training_seed')}
        rewards = np.array([episode['reward'] for episode in run['episodes']])
        costs = np.array([episode['cost'] for episode in run['episodes']])
        checkpoint_rows.append({
            **same,
            'checkpoint_file': run['file'],
            'sha256': run['sha256'],
            'simulator_steps_per_decision': run['simulator_steps_per_decision'],
            'training_simulator_steps': run['training_simulator_steps'],
            'episodes': len(rewards),
            'reward_mean': rewards.mean(), 'reward_sd': rewards.std(),
            'cost_mean': costs.mean(), 'cost_sd': costs.std(),
            'episodes_with_zero_cost': int((costs == 0).sum()),
            'episodes_replayed_exactly': sum(episode['replayed_exactly'] for episode in run['episodes']),
        })
        for episode in run['episodes']:
            episode_rows.append({**same, 'episode_seed': episode['seed'], 'reward': episode['reward'],
                                 'cost': episode['cost'], 'decisions': episode['decisions'],
                                 'replayed_exactly': episode['replayed_exactly']})
            video = run['folder'] / 'videos' / f'seed_{episode["seed"]:04d}.mp4'
            if video.exists():
                name = f'{run["task"]}/{run["variant"]}_{run["training_seed"]}_episode_{episode["seed"]}.mp4'
                make_folder((args.out / 'videos' / name).parent)
                shutil.copy(video, args.out / 'videos' / name)
                video_rows.append({'file_name': name, **same, 'episode_seed': episode['seed'],
                                   'reward': episode['reward'], 'cost': episode['cost']})

    write_csv(args.out / 'checkpoints.csv', checkpoint_rows)
    write_csv(args.out / 'episodes.csv', episode_rows)
    if video_rows:
        write_csv(args.out / 'videos' / 'metadata.csv', video_rows)

    import pyarrow.parquet as pq
    layouts, step_count = [], 0
    for task in tasks:
        of_task = [run for run in runs if run['task'] == task]
        layouts += layout_rows(of_task)
        table = steps_table(of_task)
        step_count += table.num_rows
        make_folder(args.out / 'steps')
        pq.write_table(table, args.out / 'steps' / f'{task}.parquet', compression='zstd')
    write_csv(args.out / 'layouts.csv', layouts)

    replayed = sum(row['replayed_exactly'] for row in episode_rows)
    card = (HERE / 'dataset_card.md').read_text()
    card = card.replace('{RESULTS_TABLE}', results_table(checkpoint_rows))
    card = card.replace('{REPLAYED}', f'{replayed:,} of the {len(episode_rows):,}')
    (args.out / 'README.md').write_text(card)

    size = sum(path.stat().st_size for path in args.out.rglob('*') if path.is_file())
    print(f'{len(checkpoint_rows)} checkpoints, {len(episode_rows)} episodes ({replayed} replayed exactly), {step_count} steps, '
          f'{len(video_rows)} videos, {size / 1e6:.0f} MB in {args.out}')

    if args.upload:
        # Reads the account's token from the file named by HF_TOKEN_PATH, which ms sets for every job.
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.upload, repo_type='dataset', private=True, exist_ok=True)
        api.upload_folder(folder_path=str(args.out), repo_id=args.upload, repo_type='dataset',
                          commit_message='Evaluations of the released SafeDreamer checkpoints')
        print(f'uploaded to https://huggingface.co/datasets/{args.upload}')


if __name__ == '__main__':
    main()
