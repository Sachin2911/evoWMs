"""Gathers the records of recipe-search sweeps into one folder, and uploads them if asked to.

  python3 collect.py --sweeps NAME=FOLDER [NAME=FOLDER ...] --out FOLDER [--upload ACCOUNT/NAME]

Each sweep FOLDER holds one sub-folder per run (task-0000, ...), as written by run.sh. A run's
records are copied to OUT/NAME/<task>-<label>/: its settings, every generation and validation, the
kept adapters and their evaluations, the cost network, the centres of the search and the log.
The code a run used is kept once per version in OUT/code/. summarise.py's table for each
sweep goes in OUT/NAME/summary.txt and runs.csv.

With --upload the folder is uploaded to that Hugging Face dataset under adapter_recipe_search/.
The token is read from the file named by HF_TOKEN_PATH, which ms sets for every job.
"""

import argparse
import contextlib
import io
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KEEP = ('settings.json', 'run.txt', 'validations.jsonl', 'generations.jsonl', 'finals.json', 'finished.json', 'cost_net.npz',
        'task.log', 'adapter.npz', 'episodes.jsonl')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sweeps', nargs='+', required=True)
    parser.add_argument('--out', required=True, type=pathlib.Path)
    parser.add_argument('--upload')
    parser.add_argument('--card', type=pathlib.Path, help='a page describing the records, copied to OUT/README.md')
    args = parser.parse_args()

    import summarise
    for sweep in args.sweeps:
        name, folder = sweep.split('=', 1)
        runs = sorted(path.parent for path in pathlib.Path(folder).glob('*/run.txt'))
        for run in runs:
            label, task = (run / 'run.txt').read_text().split()[:2]
            # The same label is used on several tasks, so the task is part of the folder's name.
            target = args.out / name / f'{task}-{label}'
            target.mkdir(parents=True, exist_ok=True)
            for item in KEEP:
                if (run / item).exists():
                    shutil.copy2(run / item, target / item)
            for path in list(run.glob('evaluation_*.jsonl')) + list(run.glob('adapter_*.npz')) + list(run.glob('extra_*')):
                shutil.copy2(path, target / path.name)
            if (run / 'means').is_dir():
                shutil.copytree(run / 'means', target / 'means', dirs_exist_ok=True)
            version = run / 'code' / 'VERSION'
            if version.exists():
                shutil.copy2(version, target / 'CODE_VERSION')
                words = version.read_text().split()
                code = args.out / 'code' / (words[0] if words else 'unknown')
                if not code.exists():
                    shutil.copytree(run / 'code', code, ignore=shutil.ignore_patterns('__pycache__'))
        printed = io.StringIO()
        sys.argv = ['summarise.py', str(args.out / name), '--csv', str(args.out / name / 'runs.csv'), '--curves']
        with contextlib.redirect_stdout(printed):
            summarise.main()
        (args.out / name / 'summary.txt').write_text(printed.getvalue())
        print(f'{name}: {len(runs)} runs')
    if args.card:
        shutil.copy2(args.card, args.out / 'README.md')

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.upload, repo_type='dataset', private=True, exist_ok=True)
        api.upload_folder(folder_path=str(args.out), path_in_repo='adapter_recipe_search', repo_id=args.upload,
                          repo_type='dataset', commit_message='Adapter recipe search: records')
        print(f'uploaded to https://huggingface.co/datasets/{args.upload}/tree/main/adapter_recipe_search')


if __name__ == '__main__':
    main()
