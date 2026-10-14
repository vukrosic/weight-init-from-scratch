# Weight initialization from scratch

Two networks with 30 layers learn the same pictures of handwritten digits. One gets 486 of 500 test pictures right. The other gets 41. One number differs between them: the spread of the random weights they start with, 0.177 for the first and 0.01 for the second.

This repository is the code of the video "Why a 30-Layer Network Learns Nothing". It builds the network in plain numpy, breaks it five ways, and finds the rule for the spread: the root of 2 divided by the count of numbers that go into a layer. That is the rule of He, Zhang, Ren and Sun (2015). The rule of Glorot and Bengio (2010) is here too, with the kind of network it was made for.

## Run it

```bash
pip install numpy scikit-learn
sh run_all.sh
```

That takes about 40 seconds and prints what is in [demo-output.txt](demo-output.txt). The long trainings (676 of them) are not repeated: their results are in [files/runs.jsonl](files/runs.jsonl), one line each. To train them again, delete that file and run `python3 sweep.py 0 1` (about half an hour on one laptop core).

## The pictures

`sklearn.datasets.load_digits`: 1,797 pictures of handwritten digits, 8 by 8 grey squares. They come with scikit-learn, nothing is downloaded. Every square is a number from 0 (empty) to 1 (full ink). The pictures are shuffled once and split:

- 1,097 to learn from,
- 200 to pick the step size,
- 500 for the test. Every "right of 500" in the video is counted on these.

## The network

Layers of 64 numbers. Every layer multiplies the numbers it gets by its weights and adds them up; after every layer but the last, negative sums are set to zero. The last layer gives 10 scores, one for each digit. No normalization, no skip connections, no momentum: plain learning steps on 64 pictures, 3,000 steps per training.

Every network tries 11 step sizes from 0.00001 to 1. The kept training is the one with the most right on the 200 pick pictures. So when a network fails here, an unlucky step size is not the reason.

## Words

| In the video | In the papers | In the code |
| --- | --- | --- |
| weights | weights | `net[i][0]` |
| shift | bias | `net[i][1]` |
| the start, starting weights | initialization | `start(layers, spread)` |
| spread: typical distance from zero | standard deviation | `spread` |
| count: numbers going into one sum | fan-in, n | `n_in` |
| negatives are set to zero | ReLU, rectifier | `relu` |
| smooth curve | tanh | `np.tanh` |
| size of the signal | root of the mean square of the responses | `size` |
| factor of a layer | √(n × Var(w) / 2) | `factor(spread, n_in)` |
| signal going back, move of a weight | gradient | `backward` |
| the signal dies / blows up | vanishing / exploding | — |
| step size | learning rate | `rate` |
| the rule of 2010 | Xavier or Glorot initialization | `RULES['xavier']` |
| the rule of 2015 | He or Kaiming initialization | `RULES['he']` |
| a picture's own part | — (a measure of this repository) | `own` |

## Files

| File | What it does |
| --- | --- |
| `net.py` | The network: `start`, `forward`, `backward`, `train`, and the measures `size`, `factor`, `own`. |
| `step1_network.py` | 3 layers with spread 0.01: 64 of 500 before learning, 492 after. |
| `step2_same.py` | Why the start is random: all zeros (41), all equal (160), dice (491). |
| `step3_deep.py` | More layers with spread 0.01: 41 from 5 layers on. The signal layer by layer. |
| `step4_sum.py` | Add n numbers from the dice 20,000 times: the sum has the size of the root of n. |
| `step5_layer.py` | One layer: spread × 8 × 0.71. Then 30 layers. |
| `step6_back.py` | The signal going back shrinks by the same factor. |
| `step7_dial.py` | 13 spreads at 30 layers, three throws of the dice each. Layers of 256. |
| `step8_rules.py` | The rule of 2010 and the rule of 2015, with both bends, at every depth. |
| `step9_checks.py` | Layers of mixed widths, 100 layers, the default spread of a library. |
| `sweep.py` | Every long training. Writes `files/runs.jsonl`. |
| `values.py` | Every number the video shows, written to `values.json`. |
| `tests.py` | 32 checks of the claims of the video. |
| `exercise.py` | The exercise from the end of the video, with its answer. |

## The results

Right of 500 test pictures, three throws of the dice (the starting weights and the order of the pictures change, nothing else).

| Layers | spread 0.01 | rule of 2010 (0.125) | rule of 2015 (0.177) | spread 1 |
| --- | --- | --- | --- | --- |
| 2 | 492 491 490 | 492 493 492 | 491 491 493 | 484 477 483 |
| 3 | 492 490 493 | 492 493 490 | 489 490 488 | 444 448 450 |
| 5 | 41 41 41 | 491 483 488 | 485 490 488 | 415 423 411 |
| 10 | 41 41 41 | 487 483 482 | 485 484 484 | broke |
| 20 | 41 41 41 | 466 283 366 | 483 481 473 | broke |
| 30 | 41 41 41 | 93 305 157 | 486 474 478 | broke |
| 50 | 41 41 41 | 41 39 41 | 455 233 353 | broke |
| 100 | 41 41 41 | 41 39 41 | 216 142 275 | broke |

41 is what you get by answering 7 for every picture. "broke" means the numbers grew past what the computer can hold, at the first or second learning step, with every step size.

With the smooth curve (tanh) in place of the zeroing, the rule of 2010 gets 490, 491, 487 at 30 layers. It was made for that kind of network.

## Reading the numbers

- **The count is 64 here.** A spread of 0.01 is then a factor of 0.057 per layer, and it fails from 5 layers on. The networks of 2012 to 2014 had far more numbers in each sum, so the same 0.01 lasted longer: the trouble was reported for more than 8 layers.
- **The rule of 2010 does not stall completely at 30 layers here** (93, 305, 157). In the 2015 paper it stalls on a 30-layer network for ImageNet.
- **0.177 and the rule of 2015 are the same spread to three digits** (the rule gives 0.1768). The two rows of trainings still differ (472, 486, 482 against 486, 474, 478). That is the dice, not the spread.
- **The second and third throw of the dice reuse the step size kept for the first.**
- **One training is not a result.** With 50 layers and the rule of 2015 the three throws give 455, 233 and 353.
- **The default of PyTorch** for a layer that gets 64 numbers is a uniform start with spread 0.072. Here that spread is drawn from the bell-shaped dice the rest of the code uses. 30 plain layers with it get 41, 41, 41. PyTorch networks usually do not fail like this because they have normalization layers or skip connections; a plain stack of layers does.
- **`own`** is this repository's measure, not one from the papers: take the average output over all pictures as the common pattern, remove it from every picture's output, and compare what is left with the whole.

## What this is not

It is not a reproduction of either paper. The papers use ImageNet, convolutions and far larger layers; this is 64 numbers per layer on 8 by 8 digits. What carries over is the calculation: the size of the signal is multiplied by spread × √(count / 2) in every layer that zeroes its negatives.

## Sources

- Xavier Glorot, Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." AISTATS 2010. <https://proceedings.mlr.press/v9/glorot10a.html>
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." 2015. <https://arxiv.org/abs/1502.01852>
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition." 2015. <https://arxiv.org/abs/1512.03385>

## Try it

`exercise.py`: a layer multiplies its negative sums by `keep` instead of setting them to zero. Which spread keeps the size of the signal? The video asks it for `keep = 0.25`. The answer is in the marked line of `spread_for`; cover it and try first.

In PyTorch the rule of 2015 is one line per layer:

```python
torch.nn.init.kaiming_normal_(layer.weight)
```
