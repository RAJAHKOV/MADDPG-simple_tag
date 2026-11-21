"""Export an existing MADDPG rewards.pkl without running training again."""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rewards_file', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--interval', type=int, default=100,
                        help='Console reporting interval; selects episodes, not an average')
    parser.add_argument('--preview-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be positive')
    source = args.rewards_file.resolve(strict=True)
    output = (args.output or source.with_name('training_rewards.xlsx')).resolve()
    if output.suffix.lower() != '.xlsx':
        parser.error('--output must end in .xlsx')
    if output.exists():
        parser.error(f'Output already exists; choose a new --output path: {output}')
    # Load only a rewards.pkl produced by your own training, since pickle is executable.
    with source.open('rb') as handle:
        saved = pickle.load(handle)
    rewards = saved.get('rewards') if isinstance(saved, dict) else None
    if not isinstance(rewards, dict) or not rewards:
        parser.error('Expected a non-empty rewards dictionary in the saved file')
    agents = list(rewards)
    if not all(isinstance(name, str) and name for name in agents):
        parser.error('Agent names must be non-empty strings')
    values = {name: [float(value) for value in rewards[name]] for name in agents}
    lengths = {len(series) for series in values.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 1:
        parser.error('All agents must have the same nonzero number of episodes')
    if not all(math.isfinite(value) for series in values.values() for value in series):
        parser.error('Rewards contain NaN or infinity; export stopped')
    if next(iter(lengths)) > 1048575 or len(agents) > 16382:
        parser.error('Data exceeds Excel worksheet limits')
    runtime = Path(os.environ['USERPROFILE']) / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
    node = runtime / 'bin/node.exe'
    modules = runtime / 'node_modules'
    builder = Path(__file__).with_suffix('.mjs')
    if not node.is_file() or not modules.is_dir() or not builder.is_file():
        parser.error('Required exporter files or Codex bundled spreadsheet runtime are missing')
    payload = {
        'agents': agents, 'rewards': values, 'interval': args.interval,
        'source': ' / '.join(source.parts[-4:]),
        'output': str(output),
        'previewDir': str(args.preview_dir.resolve()) if args.preview_dir else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='maddpg-export-') as temp:
        temp = Path(temp)
        data_file = temp / 'rewards.json'
        data_file.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        shutil.copy2(builder, temp / builder.name)
        # Junction works without developer mode or an extra Python package.
        powershell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
        junction = temp / 'node_modules'
        escaped_link = str(junction).replace("'", "''")
        escaped_target = str(modules).replace("'", "''")
        command = (f"New-Item -ItemType Junction -Path '{escaped_link}' "
                   f"-Target '{escaped_target}' -ErrorAction Stop | Out-Null")
        subprocess.run([str(powershell), '-NoProfile', '-NonInteractive', '-Command', command], check=True)
        try:
            subprocess.run([str(node), str(temp / builder.name), str(data_file)], check=True)
        finally:
            # Remove the junction itself before TemporaryDirectory cleans up.
            os.rmdir(junction)
    print(f'Exported {next(iter(lengths))} episodes to: {output}')


if __name__ == '__main__':
    main()
