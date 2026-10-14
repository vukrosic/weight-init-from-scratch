"""Exercise. A bend that keeps a part of every negative number.

    python exercise.py

In the video a layer sets negative numbers to zero. Here it multiplies
them by `keep` instead: 0 is the bend of the video, 1 is no bend at
all. Your part: the marked line in `spread_for`. It must return the
spread that keeps the size of the signal, for any `keep`.
"""
from net import WIDTH, forward, pictures, size, start

LAYERS = 30
KEEPS = [0.0, 0.25, 0.5, 1.0]


def bend(keep):
    """Negative numbers are multiplied by `keep`. The others stay."""
    return lambda x: x * ((x > 0) + keep * (x <= 0))


def spread_for(keep, n):
    # your part starts here
    return (2 / ((1 + keep * keep) * n)) ** 0.5
    # and ends here


def signal(spread, keep, x):
    """Size of the signal after layer 29 of 30, before training."""
    return size(forward(start(LAYERS, spread), x, bend(keep))[-2])


def answer():
    x = pictures()[2][0]
    old = (2 / WIDTH) ** 0.5
    return [(keep, signal(old, keep, x), spread_for(keep, WIDTH),
             signal(spread_for(keep, WIDTH), keep, x)) for keep in KEEPS]


if __name__ == '__main__':
    print('%d layers of %d numbers, before training' % (LAYERS, WIDTH))
    print('the pictures that go in have size 0.48')
    print('\n  keep   signal after layer 29   your     signal after layer'
          ' 29')
    print('         with spread %.4f      spread   with your spread'
          % (2 / WIDTH) ** 0.5)
    for keep, before, spread, after in answer():
        print('  %4.2f   %21.3g   %.4f   %19.3g'
              % (keep, before, spread, after))
