"""Every training of the video, one line each in files/runs.jsonl.

    python3 sweep.py 0 3 &   python3 sweep.py 1 3 &   python3 sweep.py 2 3

The two numbers: which share of the jobs, of how many shares. A job that
already has a line is skipped, so the sweep can be stopped and started
again. All 676 trainings take about half an hour of one laptop core.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from net import STEPS, forward, pictures, right, size, start, train

FILE = Path(__file__).parent / 'files' / 'runs.jsonl'
RULES = {'xavier': lambda n: (1 / n) ** 0.5, 'he': lambda n: (2 / n) ** 0.5}
DEPTHS = [2, 3, 5, 10, 20, 30, 50, 100]
RATES = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 0.1, 0.3, 1.0]
DIAL = ['0.01', '0.03', '0.05', '0.08', '0.1', '0.125', '0.15', '0.177',
        '0.2', '0.25', '0.35', '0.5', '1.0']
DIAL_DEPTH = 30
LIBRARY = '0.072'      # the spread PyTorch gives a layer of 64 by itself
LIBRARY_DEPTHS = [3, 10, 30]
BENDS = {'relu': (None, None),
         'tanh': (np.tanh, lambda out: 1 - out * out)}


def spread_of(name):
    """'he' and 'xavier' are rules, everything else is one fixed number."""
    return RULES[name] if name in RULES else float(name)


def jobs():
    """(spread, layers, bend, rate, seed) for every training, in order."""
    out = []
    for spread in ['0.01', 'xavier', 'he', '1.0']:
        for layers in DEPTHS:
            out += [(spread, layers, 'relu', r, 0) for r in RATES]
    for spread in DIAL:
        out += [(spread, DIAL_DEPTH, 'relu', r, 0) for r in RATES]
    for spread in ['xavier', 'he']:
        for layers in [10, 30, 50]:
            out += [(spread, layers, 'tanh', r, 0) for r in RATES]
    for layers in LIBRARY_DEPTHS:
        out += [(LIBRARY, layers, 'relu', r, 0) for r in RATES]
    return list(dict.fromkeys(out))


def key(run):
    return (run['spread'], run['layers'], run['bend'], run['rate'],
            run['seed'])


def done():
    if not FILE.exists():
        return []
    return [json.loads(line) for line in FILE.read_text().splitlines()]


def one(job, data):
    """Train one network and return its line for the file."""
    spread, layers, bend, rate, seed = job
    learn, pick, test = data
    bend_f, slope_f = BENDS[bend]
    net = start(layers, spread_of(spread), seed)
    signal = size(forward(net, test[0], bend_f)[-2])
    began = time.time()
    net, curve, state = train(net, learn, rate, STEPS, seed, bend_f,
                              slope_f, every=100, test=test)
    return {'spread': spread, 'layers': layers, 'bend': bend, 'rate': rate,
            'seed': seed, 'steps': STEPS, 'signal': signal, 'state': state,
            'pick': right(net, *pick, bend_f),
            'test': right(net, *test, bend_f),
            'curve': [c[1] for c in curve],
            'secs': round(time.time() - began, 1)}


def best(runs, spread, layers, bend='relu', seed=0):
    """The kept training of one cell: the most right on the pick pictures."""
    cell = [r for r in runs if (r['spread'], r['layers'], r['bend'],
                                r['seed']) == (spread, layers, bend, seed)]
    return max(cell, key=lambda r: (r['pick'], -r['rate'])) if cell else None


def kept(spread, layers, bend='relu', seed=0):
    """The kept training of one cell, read from the file."""
    return best(done(), spread, layers, bend, seed)


def seed_jobs(runs):
    """Two more throws of the dice for every kept training."""
    out = []
    for spread, layers, bend, _, _ in jobs():
        kept = best(runs, spread, layers, bend)
        if kept:
            out += [(spread, layers, bend, kept['rate'], s) for s in (1, 2)]
    return list(dict.fromkeys(out))


def main():
    share, shares = int(sys.argv[1]), int(sys.argv[2])
    data = pictures()
    FILE.parent.mkdir(exist_ok=True)
    for todo in (jobs, lambda: seed_jobs(done())):
        have = {key(r) for r in done()}
        mine = [j for i, j in enumerate(todo()) if i % shares == share]
        for job in mine:
            if job in have:
                continue
            run = one(job, data)
            with FILE.open('a') as f:
                f.write(json.dumps(run) + '\n')
            print(job, run['state'], run['pick'], run['test'], run['secs'],
                  flush=True)


if __name__ == '__main__':
    main()
