# Next-Generation Priority Ranking

Each intervention scored as `addressed_count / (complexity × compute × supervision)`. Higher is better leverage.

| rank | intervention | tags addressed | complexity | compute | supervision | priority score | expected metric movement |
|---:|---|---:|:---:|:---:|:---:|---:|---|
| 1 | more_annotation | 4057 | med | low | high | 676.2 | FUNDAMENTAL — unblocks measurement |
| 2 | more_data | 4940 | med | med | med | 617.5 | high — broad coverage |
| 3 | better_syntax | 3910 | med | med | med | 488.8 | high — addresses syntactic-depth tail |
| 4 | larger_model | 3542 | high | high | low | 393.6 | uncertain — frozen-baseline showed null at 296M-13B |
| 5 | reasoning | 4102 | high | med | high | 227.9 | uncertain — frozen baseline R2 was 0.0 |
| 6 | discourse_supervision | 685 | high | med | high | 38.1 | moderate — addresses 16% of errors |
| 7 | semantic_supervision | 205 | high | med | high | 11.4 | uncertain — none yet tested |
| 8 | better_evaluator | 4 | low | low | low | 4.0 | low — already mostly fixed for kana |
| 9 | construction_targeted_data | 7 | low | low | med | 3.5 | low — tag count is small |
| 10 | better_parser | 0 | med | med | low | 0.0 | low — tag is rare in current data |