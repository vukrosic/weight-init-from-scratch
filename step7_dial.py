"""Step 7. One dial: the spread. 30 layers, 13 positions."""
from net import WIDTH, factor, forward, pictures, size, start
from sweep import DIAL, DIAL_DEPTH, kept

print('%d layers. Right of 500 test pictures, three throws of the dice'
      % DIAL_DEPTH)
print('  spread   factor   signal after   right          what happened')
print('                    layer %d' % (DIAL_DEPTH - 1))
for name in DIAL:
    runs = [kept(name, DIAL_DEPTH, seed=s) for s in (0, 1, 2)]
    state = runs[0]['state']
    if state == 'ok':
        state = 'step size %g' % runs[0]['rate']
    print('  %6s   %6.3f   %12.2g   %3d %3d %3d    %s'
          % (name, factor(float(name), WIDTH), runs[0]['signal'],
             runs[0]['test'], runs[1]['test'], runs[2]['test'], state))

print('\nthe factor is 1 when spread x root of (%d / 2) = 1' % WIDTH)
print('  spread = root of (2 / %d) = %.4f' % (WIDTH, (2 / WIDTH) ** 0.5))

WIDE = 256
LAYERS = 10
test = pictures()[2]
print('\nis %.3f right for every network? %d layers of %d numbers'
      % (0.177, LAYERS, WIDE))
print('size of the signal after layers 1 to %d, before training'
      % (LAYERS - 1))
for spread in (0.177, (2 / WIDE) ** 0.5):
    net = start(LAYERS, spread, widths=[WIDE] * (LAYERS - 1))
    outs = forward(net, test[0])
    print('  spread %.4f (factor %.2f for layers that get %d numbers)'
          % (spread, factor(spread, WIDE), WIDE))
    print('    ' + '  '.join('%.3g' % size(o) for o in outs[1:-1]))
print('  root of (2 / %d) = %.4f' % (WIDE, (2 / WIDE) ** 0.5))
