"""Step 8. The rule of 2010 and the rule of 2015, by depth."""
from net import WIDTH, factor
from sweep import DEPTHS, RULES, kept

NAMES = {'xavier': '2010: root of (1 / n)', 'he': '2015: root of (2 / n)'}

for name, title in NAMES.items():
    spread = RULES[name](WIDTH)
    print('%s. For n = %d the spread is %.4f, the factor %.3f'
          % (title, WIDTH, spread, factor(spread, WIDTH)))

print('\nright of 500 test pictures, three throws of the dice')
print('  layers   rule of 2010     rule of 2015     signal after the'
      ' layer before last')
for layers in DEPTHS:
    row = '  %6d' % layers
    for name in NAMES:
        runs = [kept(name, layers, seed=s) for s in (0, 1, 2)]
        row += '   %3d %3d %3d   ' % tuple(r['test'] for r in runs)
    signals = [kept(name, layers)['signal'] for name in NAMES]
    print(row + '  %8.2g and %.2g' % tuple(signals))

LAYERS = 30
print('\n%d layers, the first throw of the dice: right of 500 while'
      ' it trains' % LAYERS)
print('  steps    rule of 2010   rule of 2015')
curves = [kept(name, LAYERS)['curve'] for name in NAMES]
for i in range(0, 31, 3):
    print('  %5d    %12d   %12d' % (i * 100, curves[0][i], curves[1][i]))

print('\nthe same two rules, but the layers bend with tanh')
print('(tanh keeps the negatives: no half is lost)')
print('  layers   rule of 2010     rule of 2015')
for layers in (10, 30, 50):
    row = '  %6d' % layers
    for name in NAMES:
        runs = [kept(name, layers, 'tanh', s) for s in (0, 1, 2)]
        row += '   %3d %3d %3d   ' % tuple(r['test'] for r in runs)
    print(row)
