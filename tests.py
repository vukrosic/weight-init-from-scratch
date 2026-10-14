"""Checks of the claims of the video. python tests.py

The checks read files/runs.jsonl, the table of finished trainings,
and repeat the sums of the video. Only one short training runs here.
"""
import numpy as np

from exercise import answer
from net import (WIDTH, backward, factor, forward, own, pictures, relu,
                 right, size, start, train)
from sweep import (DEPTHS, DIAL, LIBRARY, RATES, RULES, done, jobs,
                   kept)

results = []


def check(text, ok):
    results.append(bool(ok))
    print(('ok    ' if ok else 'WRONG ') + text)


def three(spread, layers, bend='relu'):
    """Right of 500 for the three throws of the dice."""
    return [kept(spread, layers, bend, s)['test'] for s in (0, 1, 2)]


learn, pick, test = pictures()
runs = done()
dice = np.random.default_rng(0)
HE, XAVIER = RULES['he'](WIDTH), RULES['xavier'](WIDTH)

# the task and the network
check('1,797 pictures: 1,097 to learn, 200 to pick, 500 test',
      (len(learn[1]), len(pick[1]), len(test[1])) == (1097, 200, 500))
check('a picture is 64 numbers from 0 to 1',
      learn[0].shape[1] == 64 and learn[0].min() == 0
      and learn[0].max() == 1)
count = sum(w.size + s.size for w, s in start(3, 0.01))
check('a network of 3 layers has 8,970 numbers', count == 8970)
count = sum(w.size + s.size for w, s in start(30, 0.01))
check('a network of 30 layers has 121,290 numbers', count == 121290)
check('the table has 676 trainings, 11 step sizes for every cell',
      len(runs) == 676 and len(jobs()) == 52 * len(RATES))

# why dice
net = start(3, 1.0)
for weights, _ in net:
    weights[:] = 0.1
net, _, _ = train(net, learn, 0.03, steps=200)
same = len(np.unique(np.round(net[0][0], 9), axis=1).T)
check('every weight 0.1 at the start: after 200 steps the 64 units of'
      ' layer 1 are still one and the same', same == 1)
sevens = int((test[1] == 7).sum())
check('41 of the 500 test pictures show a 7, and 7 is the most common'
      ' digit to learn from',
      sevens == 41 and np.bincount(learn[1]).argmax() == 7)

# one layer
sums = dice.normal(0, 1, (20000, 64)).sum(1)
check('the sum of 64 dice numbers of spread 1 has size 8 (%.2f)'
      % size(sums), abs(size(sums) - 8) < 0.15)
half = np.mean(relu(sums) ** 2) / np.mean(sums ** 2)
check('setting the negatives to zero leaves half of the average'
      ' square (%.2f)' % half, abs(half - 0.5) < 0.02)
check('factor of a layer: 0.057 for spread 0.01, 0.707 for the 2010'
      ' rule, 1 for the 2015 rule',
      abs(factor(0.01, WIDTH) - 0.0566) < 0.001
      and abs(factor(XAVIER, WIDTH) - 0.7071) < 0.001
      and abs(factor(HE, WIDTH) - 1) < 1e-9)
check('the two rules for 64 numbers: spread 0.125 and 0.177',
      XAVIER == 0.125 and abs(HE - 0.1768) < 0.0001)
outs = forward(start(30, 0.01), test[0])
said = size(outs[1]) * factor(0.01, WIDTH) ** 28
check('spread 0.01, 30 layers: the signal after layer 29 is %.1e, the'
      ' rule says %.1e' % (size(outs[29]), said),
      0.1 < size(outs[29]) / said < 10 and size(outs[29]) < 1e-35)
outs = forward(start(30, HE), test[0])
check('2015 rule, 30 layers: the signal after layer 29 is %.2f, the'
      ' pictures are %.2f' % (size(outs[29]), size(test[0])),
      0.2 < size(outs[29]) < 1)
outs = forward(start(30, 1.0), test[0])
check('spread 1, 30 layers: the signal after layer 29 is %.1e'
      % size(outs[29]), size(outs[29]) > 1e20)

# the signal going back
x, y = learn[0][:64], learn[1][:64]
backs = {}
for spread in (0.01, XAVIER, HE, 0.25):
    net = start(30, spread)
    _, moves, sizes = backward(net, forward(net, x), y)
    backs[spread] = (sizes[0], size(moves[0][0]))
check('going back through 30 layers: %.0e with spread 0.01, %.2f with'
      ' the 2015 rule, %.0e with 0.25'
      % (backs[0.01][0], backs[HE][0], backs[0.25][0]),
      backs[0.01][0] < 1e-30 and 0.05 < backs[HE][0] < 1
      and backs[0.25][0] > 1000)
check('the move of a weight in layer 1 is over 10,000 times smaller'
      ' with the 2010 rule than with the 2015 rule',
      backs[HE][1] / backs[XAVIER][1] > 10000)

# the table
check('spread 0.01 learns with 2 and 3 layers (at least 490 of 500)',
      min(three('0.01', 2) + three('0.01', 3)) >= 490)
check('spread 0.01 gets 41 of 500 from 5 layers on, for every throw',
      all(three('0.01', d) == [41, 41, 41] for d in DEPTHS[2:]))
check('spread 1 breaks from 10 layers on',
      all(kept('1.0', d)['state'].startswith('broke')
          and kept('1.0', d)['test'] == 0 for d in DEPTHS[3:]))
means = {name: np.mean(three(name, 30)) for name in DIAL}
order = sorted(means, key=means.get, reverse=True)
check('the dial at 30 layers: the best position is 0.177, then 0.2,'
      ' then 0.15', order[:3] == ['0.177', '0.2', '0.15'])
check('the dial: 41 of 500 for every spread up to 0.1, nothing at all'
      ' from 0.35 on',
      all(three(n, 30) == [41, 41, 41] for n in DIAL[:5])
      and all(three(n, 30) == [0, 0, 0] for n in DIAL[10:]))
check('30 layers: the 2010 rule gets 93, 305, 157 and the 2015 rule'
      ' 486, 474, 478',
      three('xavier', 30) == [93, 305, 157]
      and three('he', 30) == [486, 474, 478])
check('the 2015 rule gets at least 473 of 500 up to 30 layers',
      min(min(three('he', d)) for d in DEPTHS[:6]) >= 473)
check('up to 10 layers the two rules are within 10 pictures',
      all(abs(np.mean(three('he', d)) - np.mean(three('xavier', d)))
          < 10 for d in DEPTHS[:4]))
check('the 2010 rule gets at most 41 with 50 and 100 layers',
      max(three('xavier', 50) + three('xavier', 100)) <= 41)
check('with tanh both rules get at least 484 at 10 and 30 layers',
      min(min(three(r, d, 'tanh')) for r in RULES for d in (10, 30))
      >= 484)

# what the rule cannot do
check('100 layers with the 2015 rule: 216, 142, 275',
      three('he', 100) == [216, 142, 275])
shares = [own(forward(start(100, RULES['he'], s), test[0])[99])
          for s in (0, 1, 2)]
check('after layer 99 a picture\'s own share of the signal is under'
      ' 0.04 (it is %.2f in the pictures)' % own(test[0]),
      max(shares) < 0.04 and own(test[0]) > 0.5)
wide = [256] * 9
outs = forward(start(10, 0.177, widths=wide), test[0])
grew = size(outs[9]) / size(outs[1])
outs = forward(start(10, RULES['he'], widths=wide), test[0])
stay = size(outs[9]) / size(outs[1])
check('layers of 256 numbers: spread 0.177 grows the signal %d times'
      ' in 8 layers, the rule keeps it (x %.2f)' % (grew, stay),
      100 < grew < 500 and 0.5 < stay < 2)
check('the spread a library gives by itself, 0.072: 3 layers learn,'
      ' 30 layers get 41 of 500',
      abs(1 / (3 * WIDTH) ** 0.5 - float(LIBRARY)) < 0.001
      and min(three(LIBRARY, 3)) >= 489
      and three(LIBRARY, 30) == [41, 41, 41])

# the exercise
rows = answer()
check('exercise: for keep = 0 the spread is the 2015 rule, for keep ='
      ' 1 the 2010 rule',
      abs(rows[0][2] - HE) < 1e-9 and abs(rows[-1][2] - XAVIER) < 1e-9)
check('exercise: with the right spread the signal after layer 29'
      ' stays between 0.2 and 1',
      all(0.2 < row[3] < 1 for row in rows))

print('\n%d of %d checks ok' % (sum(results), len(results)))
assert all(results)
