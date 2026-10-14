"""Step 3. The same start, spread 0.01, with more and more layers."""
import numpy as np

from net import forward, pictures, size, start, train
from sweep import DEPTHS, kept

learn, pick, test = pictures()
SPREAD = 0.01

print('dice with spread %g, best of 11 step sizes for every depth'
      % SPREAD)
print('  layers   right of %d   step size kept' % len(test[1]))
for layers in DEPTHS:
    run = kept(str(SPREAD), layers)
    print('  %6d   %12d   %14g' % (layers, run['test'], run['rate']))

print('\ninside the network before training: size of the signal')
print('(the %d test pictures go in; their own size is %.2f)'
      % (len(test[1]), size(test[0])))
for layers in (3, 5, 30):
    outs = forward(start(layers, SPREAD), test[0])
    shown = [i for i in (1, 2, 3, 4, 10, 20, 29) if i < layers]
    print('  %2d layers: ' % layers + '  '.join(
        'after %d: %.2g' % (i, size(outs[i])) for i in shown))

LAYERS = 5
net = start(LAYERS, SPREAD)
print('\n%d layers, before training: scores for two test pictures'
      % LAYERS)
print('(a %d and a %d)' % (test[1][0], test[1][1]))
for scores in forward(net, test[0][:2])[-1]:
    print('  ' + ' '.join('%+.1e' % v for v in scores[:5]) + ' ...')

run = kept(str(SPREAD), LAYERS)
net, _, _ = train(net, learn, run['rate'])
answers = forward(net, test[0])[-1].argmax(1)
digit = int(np.bincount(answers).argmax())
print('\nafter %d layers were trained, what it answers:' % LAYERS)
print('  the digit %d for %d of %d test pictures'
      % (digit, (answers == digit).sum(), len(answers)))
print('  test pictures that show a %d: %d'
      % (digit, (test[1] == digit).sum()))
print('  pictures to learn from that show a %d: %d, more than any'
      ' other digit' % (digit, (learn[1] == digit).sum()))
