"""A plain network for 8 by 8 pictures of digits. Numpy only.

A picture is 64 numbers. Every layer is a table of weights: it turns
the numbers it gets into new numbers. After every layer but the last,
negative numbers are set to zero. The last layer gives 10 scores, one
for each digit.
"""
import numpy as np
from sklearn.datasets import load_digits

PIXELS = 64      # 8 by 8 grey squares
WIDTH = 64       # numbers that come out of every layer but the last
DIGITS = 10
BATCH = 64       # pictures in one learning step
STEPS = 3000     # learning steps in one training


def pictures():
    """1,797 pictures, shuffled: 1,097 to learn, 200 to pick, 500 test."""
    data = load_digits()
    order = np.random.default_rng(0).permutation(len(data.target))
    x = data.data[order] / 16.0          # every grey square: 0 to 1
    y = data.target[order]
    return (x[:1097], y[:1097]), (x[1097:1297], y[1297 - 200:1297]), \
        (x[1297:], y[1297:])


def size(numbers):
    """Typical distance from zero: square, average, take the root."""
    return float(np.sqrt(np.mean(np.square(numbers))))


def start(layers, spread, seed=0, widths=None):
    """Random starting weights. `spread` is their typical distance from 0.

    `spread` is one number for every layer, or a rule: a function that
    gets the count of numbers going into a layer and returns the spread.
    `widths` lists how many numbers come out of each layer but the last.
    """
    dice = np.random.default_rng(seed)
    widths = [PIXELS] + (widths or [WIDTH] * (layers - 1)) + [DIGITS]
    net = []
    for n_in, n_out in zip(widths[:-1], widths[1:]):
        s = spread(n_in) if callable(spread) else spread
        net.append([dice.normal(0, 1, (n_in, n_out)) * s, np.zeros(n_out)])
    return net


def factor(spread, n_in):
    """What one layer does to the size of the signal, by the rule.

    The sum of n_in products is the root of n_in times as big as one
    product. Setting the negatives to zero halves the average square.
    """
    return spread * (n_in / 2) ** 0.5


def forward(net, x, bend=None):
    """Push pictures through. Returns the input and what each layer gave."""
    bend = bend or relu
    outs = [x]
    for i, (weights, shift) in enumerate(net):
        x = x @ weights + shift
        if i < len(net) - 1:
            x = bend(x)
        outs.append(x)
    return outs


def relu(x):
    """Negative numbers become zero. The others stay."""
    return np.maximum(x, 0)


def backward(net, outs, labels, slope=None):
    """The loss, the move for every weight, and the signal going back.

    The signal going back says, for every number a layer gave, how the
    loss changes when that number grows. `sizes[i]` is its size at the
    output of layer i.
    """
    slope = slope or (lambda out: out > 0)
    n = len(labels)
    rows = np.arange(n)
    scores = outs[-1] - outs[-1].max(1, keepdims=True)
    log_chance = scores - np.log(np.exp(scores).sum(1, keepdims=True))
    loss = float(-log_chance[rows, labels].mean())
    back = np.exp(log_chance)
    back[rows, labels] -= 1
    moves, sizes = [], []
    for i in reversed(range(len(net))):
        sizes.append(size(back))
        moves.append((outs[i].T @ back / n, back.sum(0) / n))
        back = back @ net[i][0].T
        if i > 0:
            back = back * slope(outs[i])
    return loss, moves[::-1], sizes[::-1]


def own(out):
    """The share of a layer's signal that is a picture's own.

    The common pattern is the average of what the layer gives over all
    pictures. From every picture's output we take away as much of that
    pattern as fits. What is left is the picture's own.
    """
    pattern = out.mean(0)
    pattern = pattern / np.sqrt((pattern ** 2).sum())
    rest = out - np.outer(out @ pattern, pattern)
    return size(rest) / size(out)


def right(net, x, y, bend=None):
    """How many pictures the network names correctly."""
    with np.errstate(all='ignore'):
        scores = forward(net, x, bend)[-1]
    if not np.isfinite(scores).all():
        return 0
    return int((scores.argmax(1) == y).sum())


def train(net, data, rate, steps=STEPS, seed=0, bend=None, slope=None,
          every=0, test=None):
    """Plain learning steps. Returns (net, curve). Stops if numbers break.

    One step: take 64 pictures, find the move for every weight, and move
    every weight against it by `rate` times the move.
    """
    x, y = data
    dice = np.random.default_rng(1000 + seed)
    curve = []
    with np.errstate(all='ignore'):
        for step in range(steps):
            if every and step % every == 0:
                curve.append((step, right(net, *test, bend)))
            pick = dice.integers(0, len(y), BATCH)
            outs = forward(net, x[pick], bend)
            loss, moves, _ = backward(net, outs, y[pick], slope)
            if not np.isfinite(loss):
                return net, curve, 'broke at step %d' % step
            for (weights, shift), (d_w, d_s) in zip(net, moves):
                weights -= rate * d_w
                shift -= rate * d_s
    if every:
        curve.append((steps, right(net, *test, bend)))
    return net, curve, 'ok'
