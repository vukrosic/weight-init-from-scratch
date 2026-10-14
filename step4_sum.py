"""Step 4. Add n random numbers: how big is the sum?"""
import numpy as np

from net import size

dice = np.random.default_rng(0)
TRIES = 20000

print('one try: 4 numbers from the dice, spread 1')
four = dice.normal(0, 1, 4)
print('  ' + '  '.join('%+.2f' % v for v in four)
      + '   sum %+.2f' % four.sum())
print('  if all four had the same sign the sum would be near 4')

print('\n%s tries for every count' % format(TRIES, ','))
print('  count   size of the sum   root of the count')
for n in [1, 4, 16, 64, 256, 1024]:
    sums = dice.normal(0, 1, (TRIES, n)).sum(1)
    print('  %5d   %15.2f   %17.2f' % (n, size(sums), n ** 0.5))
print('4 times the count: the sum is 2 times as big, not 4 times')

print('\nthe same with numbers of spread 0.5 and 64 of them')
sums = dice.normal(0, 0.5, (TRIES, 64)).sum(1)
print('  size of the sum: %.2f   (0.5 x 8 = %.2f)' % (size(sums), 0.5 * 8))
