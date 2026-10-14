"""Step 5. One layer changes the size of the signal. 30 layers repeat it."""
import numpy as np

from net import WIDTH, factor, forward, pictures, relu, size, start

learn, pick, test = pictures()
dice = np.random.default_rng(0)
SPREADS = [0.01, 0.125, 0.177, 0.25, 1.0]

print('one layer: 64 numbers of size 1 go in, 64 come out')
signal = dice.normal(0, 1, (2000, WIDTH))
print('  spread   sum      after zeroing   by the rule')
print('           (x 8)    the negatives   spread x 8 x 0.71')
for spread in SPREADS:
    weights = dice.normal(0, spread, (WIDTH, WIDTH))
    summed = signal @ weights
    print('  %6.3f   %6.3f   %13.3f   %17.3f'
          % (spread, size(summed), size(relu(summed)),
             factor(spread, WIDTH)))
half = np.mean(relu(signal) ** 2) / np.mean(signal ** 2)
print('zeroing the negatives leaves %.2f of the average square,'
      % half)
print('so the size is multiplied by %.2f' % half ** 0.5)

LAYERS = 30
print('\n%d layers, the pictures go in. Size of the signal:' % LAYERS)
print('  spread   factor   after layer 1   layer 10    layer 20    '
      'layer %d    rule for layer %d' % (LAYERS - 1, LAYERS - 1))
for spread in SPREADS:
    outs = forward(start(LAYERS, spread), test[0])
    said = size(outs[1]) * factor(spread, WIDTH) ** (LAYERS - 2)
    print('  %6.3f   %6.3f   %13.3g   %8.3g   %9.3g   %9.3g   %9.3g'
          % (spread, factor(spread, WIDTH), size(outs[1]), size(outs[10]),
             size(outs[20]), size(outs[LAYERS - 1]), said))
print('the pictures themselves have size %.2f' % size(test[0]))

print('\nwhat the 10 scores look like for the first two test pictures')
for spread in SPREADS:
    scores = forward(start(LAYERS, spread), test[0][:2])[-1]
    print('  spread %5.3f   picture 1: %s' % (spread, ' '.join(
        '%9.2g' % v for v in scores[0][:4])) + ' ...')
    print('                 picture 2: %s' % ' '.join(
        '%9.2g' % v for v in scores[1][:4]) + ' ...')
