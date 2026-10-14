"""Every number the film shows, written to values.json.

    python3 values.py

Reads files/runs.jsonl (made by sweep.py) and repeats the short
measurements of the step files. Takes about 40 seconds.
"""
import json
from pathlib import Path

import numpy as np

from exercise import KEEPS, signal as keep_signal, spread_for
from net import (WIDTH, backward, factor, forward, own, pictures, relu,
                 right, size, start, train)
from sweep import (DEPTHS, DIAL, DIAL_DEPTH, LIBRARY, LIBRARY_DEPTHS, RATES,
                   RULES, done, kept)

learn, pick, test = pictures()
RUNS = done()
SEEDS = (0, 1, 2)


def r3(v):
    """Three digits that count, as a plain float."""
    return float('%.3g' % v)


def cell(spread, layers, bend='relu'):
    """One cell of a table: right of 500 for three throws of the dice."""
    runs = [kept(spread, layers, bend, s) for s in SEEDS]
    return {'right': [r['test'] for r in runs], 'rate': runs[0]['rate'],
            'state': runs[0]['state'], 'signal': r3(runs[0]['signal'])}


def by_rate(spread, layers):
    """What every one of the 11 step sizes did, dice throw 0."""
    rows = [r for r in RUNS if (r['spread'], r['layers'], r['bend'],
                                r['seed']) == (spread, layers, 'relu', 0)]
    rows.sort(key=lambda r: r['rate'])
    return [{'rate': r['rate'], 'pick': r['pick'], 'test': r['test'],
             'state': r['state']} for r in rows]


def sample():
    """One test picture of every digit, for the drawings."""
    out = []
    for digit in range(10):
        i = int(np.flatnonzero(test[1] == digit)[0])
        out.append({'digit': digit,
                    'squares': [round(float(v), 2) for v in test[0][i]]})
    return out


def scores(spread, layers, index):
    """The 10 scores of one test picture, before any learning."""
    out = forward(start(layers, spread), test[0][index:index + 1])[-1][0]
    return [r3(v) for v in out]


def same():
    """Step 2: three starts for 3 layers, the best of the 11 step sizes."""
    def constant(value):
        net = start(3, 1.0)
        for weights, _ in net:
            weights[:] = value
        return net
    starts = [('zero', lambda: constant(0.0)),
              ('equal', lambda: constant(0.1)),
              ('dice', lambda: start(3, 0.1))]
    out = {}
    for name, make in starts:
        tries = []
        for rate in RATES:
            net, _, _ = train(make(), learn, rate)
            tries.append((right(net, *pick), rate, net))
        _, rate, net = max(tries, key=lambda t: (t[0], -t[1]))
        with np.errstate(all='ignore'):
            answers = np.unique(forward(net, test[0])[-1].argmax(1))
        moved = sum(int((w != b).sum())
                    for (w, _), (b, _) in zip(net, make()))
        units = len(np.unique(np.round(net[0][0], 9), axis=1).T)
        out[name] = {'right': right(net, *test), 'rate': rate,
                     'moved': moved, 'units': units,
                     'weights': sum(w.size for w, _ in net),
                     'answers': [int(a) for a in answers]}
    return out


def sums():
    """Step 4: add n numbers from the dice, 20,000 times."""
    dice = np.random.default_rng(0)
    four = dice.normal(0, 1, 4)
    rows = []
    for n in [1, 4, 16, 64, 256, 1024]:
        got = dice.normal(0, 1, (20000, n)).sum(1)
        rows.append({'count': n, 'size': round(size(got), 2)})
    return {'four': [round(float(v), 2) for v in four],
            'sum': round(float(four.sum()), 2), 'rows': rows}


def one_layer():
    """Step 5: 64 numbers of size 1 through one layer."""
    dice = np.random.default_rng(0)
    signal = dice.normal(0, 1, (2000, WIDTH))
    rows = []
    for spread in [0.01, 0.125, 0.177, 0.25, 1.0]:
        summed = signal @ dice.normal(0, spread, (WIDTH, WIDTH))
        rows.append({'spread': spread, 'sum': round(size(summed), 3),
                     'zeroed': round(size(relu(summed)), 3),
                     'rule': round(factor(spread, WIDTH), 3)})
    half = np.mean(relu(signal) ** 2) / np.mean(signal ** 2)
    return {'rows': rows, 'half': round(float(half), 2),
            'root': round(float(half) ** 0.5, 2)}


def through(layers, spread, seed=0, widths=None):
    """Size of the signal: the pictures, then after every layer."""
    outs = forward(start(layers, spread, seed, widths), test[0])
    return [r3(size(o)) for o in outs]


def going_back(spread, layers=30):
    """Step 6: the signal going back and the move of one weight."""
    net = start(layers, spread)
    x, y = learn[0][:64], learn[1][:64]
    _, moves, sizes = backward(net, forward(net, x), y)
    return {'back': [r3(s) for s in sizes],
            'move': [r3(size(m[0])) for m in moves],
            'share': r3(0.02 * size(moves[0][0]) / size(net[0][0]))}


def hundred():
    """Step 9: 100 layers with the rule of 2015, before any learning."""
    out = []
    for seed in SEEDS:
        outs = forward(start(100, RULES['he'], seed), test[0])
        out.append({'size': [r3(size(o)) for o in outs[:-1]],
                    'own': [round(own(o), 3) for o in outs[:-1]],
                    'alive': int((outs[99].max(0) > 0).sum())})
    return out


def main():
    table = {s: {str(d): cell(s, d) for d in DEPTHS}
             for s in ['0.01', 'xavier', 'he', '1.0']}
    trained, _, _ = train(start(5, 0.01), learn,
                          kept('0.01', 5)['rate'])
    seven = forward(trained, test[0])[-1].argmax(1)
    values = {
        'counts': {'learn': len(learn[1]), 'pick': len(pick[1]),
                   'test': len(test[1]), 'rates': RATES,
                   'weights3': 64 * 64 + 64 * 64 + 64 * 10},
        'sample': sample(),
        'picture_size': round(size(test[0]), 2),
        'table': table,
        'rates': {'small3': by_rate('0.01', 3),
                  'small30': by_rate('0.01', 30),
                  'he30': by_rate('he', 30), 'one30': by_rate('1.0', 30)},
        'curves': {'small3': kept('0.01', 3)['curve'],
                   'small30': kept('0.01', 30)['curve'],
                   'xavier30': kept('xavier', 30)['curve'],
                   'he30': kept('he', 30)['curve']},
        'before': {'right3': right(start(3, 0.01), *test),
                   'scores3': scores(0.01, 3, 0),
                   'scores30': {s: [scores(float(s), 30, 0),
                                    scores(float(s), 30, 1)]
                                for s in ['0.01', '0.177', '1.0']}},
        'same': same(),
        'sevens': {'answers': int((seven == 7).sum()),
                   'test': int((test[1] == 7).sum()),
                   'learn': int((learn[1] == 7).sum())},
        'signal': {s: through(30, float(s))
                   for s in ['0.01', '0.125', '0.177', '0.25', '1.0']},
        'sums': sums(),
        'one_layer': one_layer(),
        'back': {s: going_back(float(s))
                 for s in ['0.01', '0.125', '0.177', '0.25']},
        'dial': [dict(cell(s, DIAL_DEPTH), spread=float(s),
                      factor=round(factor(float(s), WIDTH), 3))
                 for s in DIAL],
        'wide': {'old': through(10, 0.177, widths=[256] * 9),
                 'rule': through(10, (2 / 256) ** 0.5, widths=[256] * 9),
                 'spread': round((2 / 256) ** 0.5, 4)},
        'rules': {'xavier': round(RULES['xavier'](WIDTH), 4),
                  'he': round(RULES['he'](WIDTH), 4)},
        'tanh': {s: {str(d): cell(s, d, 'tanh') for d in [10, 30, 50]}
                 for s in ['xavier', 'he']},
        'hundred': hundred(),
        'library': {'spread': float(LIBRARY),
                    'factor': round(factor(float(LIBRARY), WIDTH), 2),
                    'rows': {str(d): cell(LIBRARY, d)
                             for d in LIBRARY_DEPTHS}},
        'exercise': [{'keep': k, 'spread': round(spread_for(k, WIDTH), 4),
                      'old': r3(keep_signal(0.1768, k, test[0])),
                      'new': r3(keep_signal(spread_for(k, WIDTH), k,
                                            test[0]))} for k in KEEPS],
    }
    Path(__file__).with_name('values.json').write_text(
        json.dumps(values, indent=1))
    print('values.json written,', len(json.dumps(values)), 'characters')


if __name__ == '__main__':
    main()
