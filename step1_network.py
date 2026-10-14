"""Step 1. The task, the network, and one training that works."""
from net import (DIGITS, PIXELS, STEPS, WIDTH, forward, pictures, right,
                 start, train)
from sweep import kept

learn, pick, test = pictures()
print('pictures: %d to learn, %d to pick the step size, %d test'
      % (len(learn[1]), len(pick[1]), len(test[1])))
print('one picture: %d grey squares, each from 0 to 1' % PIXELS)
print('the first test picture shows a %d:' % test[1][0])
for row in test[0][0].reshape(8, 8):
    print('   ' + ''.join(' .:*#'[min(4, int(v * 5))] * 2 for v in row))

LAYERS = 3
net = start(LAYERS, 0.01)
count = sum(w.size + s.size for w, s in net)
print('\na network of %d layers' % LAYERS)
for i, (w, s) in enumerate(net):
    print('  layer %d: %2d numbers in, %2d out, %5d weights and %2d shifts'
          % (i + 1, w.shape[0], w.shape[1], w.size, s.size))
print('  numbers that training can move: %s' % format(count, ','))

scores = forward(net, test[0][:1])[-1][0]
print('\nbefore training, the 10 scores for the first test picture:')
print('  ' + ' '.join('%+.5f' % v for v in scores))
print('  right on the test pictures: %d of %d'
      % (right(net, *test), len(test[1])))

rate = kept('0.01', LAYERS)['rate']
net, curve, state = train(net, learn, rate, every=500, test=test)
print('\n%s learning steps of 64 pictures, step size %g'
      % (format(STEPS, ','), rate))
for step, n in curve:
    print('  after %5d steps: %3d of %d right' % (step, n, len(test[1])))
scores = forward(net, test[0][:1])[-1][0]
print('after training, the 10 scores for the first test picture:')
print('  ' + ' '.join('%+.1f' % v for v in scores))
print('  the highest score is for the digit %d' % scores.argmax())
assert WIDTH == PIXELS and DIGITS == 10
