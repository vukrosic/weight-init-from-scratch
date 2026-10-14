"""Step 9. Checks: layers of other widths, and what the rule cannot do."""
from net import WIDTH, factor, forward, own, pictures, size, start
from sweep import LIBRARY, LIBRARY_DEPTHS, RULES, kept

learn, pick, test = pictures()
x = test[0]

WIDTHS = [256, 16, 128, 32, 256, 16, 128, 32, 64]
LAYERS = len(WIDTHS) + 1
print('%d layers of different widths: %s'
      % (LAYERS, ' '.join(map(str, WIDTHS))))
print('size of the signal after every layer but the last')
STARTS = [('one spread, 0.177', 0.177), ('rule of 2015', RULES['he'])]
for name, spread in STARTS:
    outs = forward(start(LAYERS, spread, widths=WIDTHS), x)
    print('  %-18s' % name + ' '.join('%5.2f' % size(o)
                                      for o in outs[1:-1]))
print('  the rule gives every layer its own spread: '
      + ' '.join('%.3f' % RULES['he'](n) for n in [64] + WIDTHS[:3])
      + ' ...')


LAYERS = 100
print('\n%d layers of 64 with the rule of 2015: right of 500 by dice'
      % LAYERS)
print('  ' + ' '.join(str(kept('he', LAYERS, seed=s)['test'])
                      for s in (0, 1, 2)))
print('before training, for three throws of the dice')
SHOWN = [0, 1, 10, 30, 50, 99]
print('  after layer:                 '
      + ' '.join('%6d' % i for i in SHOWN))
for seed in (0, 1, 2):
    outs = forward(start(LAYERS, RULES['he'], seed), x)
    print('  size of the signal:          '
          + ' '.join('%6.2f' % size(outs[i]) for i in SHOWN))
    print('  share that is a picture\'s own:'
          + ' '.join('%6.3f' % own(outs[i]) for i in SHOWN))
    alive = [int((outs[i].max(0) > 0).sum()) for i in SHOWN]
    print('  numbers that are ever above 0:'
          + ' '.join('%6d' % n for n in alive))

print('\nthe same for 30 layers, after layer 29')
for seed in (0, 1, 2):
    out = forward(start(30, RULES['he'], seed), x)[29]
    print('  size %.2f   a picture\'s own %.3f   ever above 0: %d'
          % (size(out), own(out), (out.max(0) > 0).sum()))

print('\nthe spread a library gives a layer when nobody sets it')
print('PyTorch, a layer that gets n numbers: root of (1 / (3 n))')
print('for n = %d that is %.4f, a factor of %.2f'
      % (WIDTH, 1 / (3 * WIDTH) ** 0.5, factor(float(LIBRARY), WIDTH)))
print('  layers   right of 500, three throws   rule of 2015')
for layers in LIBRARY_DEPTHS:
    runs = [kept(LIBRARY, layers, seed=s)['test'] for s in (0, 1, 2)]
    rule = [kept('he', layers, seed=s)['test'] for s in (0, 1, 2)]
    print('  %6d   %3d %3d %3d                 %3d %3d %3d'
          % (layers, *runs, *rule))
