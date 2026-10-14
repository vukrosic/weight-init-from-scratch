"""Step 2. Why the start is random: zeros, one number, dice."""
import numpy as np

from net import forward, pictures, right, start, train
from sweep import RATES

learn, pick, test = pictures()
LAYERS = 3


def constant(value):
    """Every weight gets the same number."""
    net = start(LAYERS, 1.0)
    for weights, _ in net:
        weights[:] = value
    return net


def different(table):
    """How many units of a layer differ from each other."""
    return len(np.unique(np.round(table, 9), axis=1).T)


STARTS = [('every weight 0', lambda: constant(0.0)),
          ('every weight 0.1', lambda: constant(0.1)),
          ('dice, spread 0.1', lambda: start(LAYERS, 0.1))]
for name, make in STARTS:
    tries = []
    for rate in RATES:
        net, _, _ = train(make(), learn, rate)
        tries.append((right(net, *pick), rate, net))
    _, rate, net = max(tries, key=lambda t: (t[0], -t[1]))
    moved = sum(int((w != b).sum())
                for (w, _), (b, _) in zip(net, make()))
    total = sum(w.size for w, _ in net)
    answers = np.unique(forward(net, test[0])[-1].argmax(1))
    print(name + ' (best of %d step sizes: %g)' % (len(RATES), rate))
    print('  weights that moved: %s of %s'
          % (format(moved, ','), format(total, ',')))
    for i in (0, 1):
        print('  units of layer %d that differ from each other: %d of %d'
              % (i + 1, different(net[i][0]), net[i][0].shape[1]))
    print('  digits it ever answers: %s' % ' '.join(map(str, answers)))
    print('  right on the test pictures: %d of %d'
          % (right(net, *test), len(test[1])))
