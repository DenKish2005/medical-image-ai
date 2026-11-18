"""Status-aware training weights; does not alter clinical extraction rules.

Positive/uncertain/negative alpha controls the increment above the ordinary
weight 1. Overlapping spans take the maximum weight, never a sum. Spans
crossing the truncated target are dropped, matching clinical_spans.py.
"""
from collections import Counter
import math

import clinical_spans as cs

STATUSES = ('positive', 'uncertain', 'negative')


def weights_from_mentions(n_targets, mentions, alphas):
    if n_targets < 0:
        raise ValueError('n_targets must not be negative')
    if set(alphas) != set(STATUSES):
        raise ValueError('Expected positive, uncertain and negative alpha values')
    if any(not math.isfinite(float(a)) or a < 0 for a in alphas.values()):
        raise ValueError('Alpha values must be finite and nonnegative')
    weights = [1.0] * n_targets
    touched = [set() for _ in range(n_targets)]
    recognized, included, truncated = Counter(), Counter(), Counter()
    for m in mentions:
        status = m['status']
        if status not in alphas:
            raise ValueError(f'Unknown mention status: {status}')
        start, end = int(m['start']), int(m['end'])
        if start < 0 or end < start:
            raise ValueError('Invalid mention span')
        recognized[status] += 1
        if end > n_targets:
            truncated[status] += 1
            continue
        included[status] += 1
        for j in range(start, end):
            touched[j].add(status)
            weights[j] = max(weights[j], 1.0 + float(alphas[status]))
    info = {
        'n_targets': n_targets,
        'mentions': len(mentions),
        'recognized_by_status': {s: recognized[s] for s in STATUSES},
        'included_by_status': {s: included[s] for s in STATUSES},
        'truncated_by_status': {s: truncated[s] for s in STATUSES},
        'truncated_spans': sum(truncated.values()),
        'upweighted_tokens': sum(w > 1.0 for w in weights),
        'touched_tokens_by_status': {s: sum(s in v for v in touched) for s in STATUSES},
        'tokens_with_overlapping_statuses': sum(len(v) > 1 for v in touched),
        'tokens_at_max_weight': {s: sum((s in touched[j]) and (weights[j] > 1.0) and
                                       weights[j] == 1.0 + float(alphas[s])
                                       for j in range(n_targets)) for s in STATUSES},
    }
    return weights, info


def token_weights(words, n_targets, alpha_positive, alpha_uncertain, alpha_negative):
    alphas = {'positive': alpha_positive, 'uncertain': alpha_uncertain, 'negative': alpha_negative}
    return weights_from_mentions(n_targets, cs.find_mentions(words), alphas)


def self_test():
    # Keep the original extractor and its published polarity/scope checks.
    if cs.run_polarity_tests(verbose=True):
        raise AssertionError('Underlying clinical span polarity tests failed')
    # Equal alphas must produce precisely the old weighting algorithm.
    for text, _, _, _ in cs.POLARITY_TESTS:
        words = text.split()
        for n_targets in (len(words) + 1, max(1, len(words) // 2)):
            new, _ = token_weights(words, n_targets, 2, 2, 2)
            old, _ = cs.token_weights(words, n_targets, 2)
            if new != old:
                raise AssertionError(f'Uniform weighting differs for: {text}')
    # Examples verify finding, negation and uncertainty-cue weighting.
    examples = [
        ('small left pleural effusion .', {'small': 5, 'effusion': 5}),
        ('no pleural effusion .', {'no': 3, 'effusion': 3}),
        ('cannot exclude edema .', {'cannot': 5, 'exclude': 5, 'edema': 5}),
    ]
    for text, checks in examples:
        words = text.split()
        w, _ = token_weights(words, len(words) + 1, 4, 4, 2)
        for word, expected in checks.items():
            if w[words.index(word)] != expected:
                raise AssertionError(f'Unexpected weight for {word!r} in {text!r}: {w}')
        if w[-1] != 1.0:
            raise AssertionError('EOS must retain ordinary weight')
    # Synthetic overlap is max, not sum; truncation drops the entire cut span.
    mentions = [{'status': 'negative', 'start': 0, 'end': 2},
                {'status': 'positive', 'start': 1, 'end': 3},
                {'status': 'uncertain', 'start': 2, 'end': 5}]
    w, info = weights_from_mentions(3, mentions, {'positive': 4, 'uncertain': 4, 'negative': 2})
    if w != [3.0, 5.0, 5.0] or info['truncated_by_status']['uncertain'] != 1:
        raise AssertionError('Overlap or truncated-span handling failed')
    print('Status-weight self-test passed: baseline equivalence, cues, overlap, truncation.')


if __name__ == '__main__':
    self_test()
