def saved_keystrokes(gt: str, suggestion: str) -> int:
    incr = 1
    score = 0
    if len(suggestion) > len(gt):
        gt = list(gt) + [None] * (len(suggestion) - len(gt))

    for c1, c2 in zip(gt, suggestion):
        if c1 != c2 and incr > 0:
            incr = -incr
        score += incr

    return score
