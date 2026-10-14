"""Step 6. Training sends a signal back through the same weights."""
from net import WIDTH, backward, factor, forward, pictures, size, start

learn, pick, test = pictures()
LAYERS = 30
x, y = learn[0][:64], learn[1][:64]
SPREADS = [0.01, 0.125, 0.177, 0.25]

print('%d layers, before the first learning step, 64 pictures' % LAYERS)
print('\nsize of the signal going back, at the output of a layer')
print('  spread   factor   layer %d   layer 20   layer 10   layer 1'
      % LAYERS)
kept = {}
for spread in SPREADS:
    net = start(LAYERS, spread)
    loss, moves, sizes = backward(net, forward(net, x), y)
    kept[spread] = (net, moves)
    print('  %6.3f   %6.3f   %8.2g   %8.2g   %8.2g   %8.2g'
          % (spread, factor(spread, WIDTH), sizes[LAYERS - 1], sizes[19],
             sizes[9], sizes[0]))

print('\nsize of the move for one weight (before the step size)')
print('  spread   layer 1    layer 15   layer %d   weight itself'
      % LAYERS)
for spread in SPREADS:
    net, moves = kept[spread]
    print('  %6.3f   %8.2g   %8.2g   %8.2g   %13.3f'
          % (spread, size(moves[0][0]), size(moves[14][0]),
             size(moves[-1][0]), size(net[0][0])))

RATE = 0.02
print('\none learning step with step size %g moves a weight of layer 1'
      % RATE)
print('by this share of its own size')
for spread in SPREADS:
    net, moves = kept[spread]
    share = RATE * size(moves[0][0]) / size(net[0][0])
    print('  spread %5.3f   %.2g' % (spread, share))
