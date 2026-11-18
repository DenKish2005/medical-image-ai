"""Rule-based clinical span annotation (training weights) and concept labelling (evaluation).

Operates on R2Gen's cleaned IU X-Ray word sequence (tokenizer.clean_report(report).split()),
so spans align 1:1 with the decoder's target tokens.

A *mention* = (concept, status, start, end) where status in {'positive','negative','uncertain'}.
A *span* covers the finding phrase together with its negation/uncertainty cue and adjacent
anatomy/laterality/severity modifiers, so that e.g. "no pleural effusion" or
"cannot exclude edema" are weighted as a unit (technical note, Sec. 4.3).

This is a documented, finite heuristic (a NegEx-style rule set), NOT clinical ground truth.
Freeze RULES_VERSION and the lexicons before test evaluation and audit samples manually.
For publication-grade clinical metrics also run CheXbert / RadGraph on the generated reports.
"""
import hashlib
import inspect
import sys

RULES_VERSION = 'ta-spans-1.0'

# concept -> list of word-sequence patterns (exact cleaned tokens).  Longest match wins.
CONCEPTS = {
    'cardiomegaly': ['cardiomegaly', 'cardiac enlargement', 'enlarged heart', 'enlarged cardiac silhouette',
                     'heart size is enlarged', 'heart is enlarged', 'heart is mildly enlarged',
                     'heart size is mildly enlarged', 'heart size is borderline', 'heart is borderline',
                     'borderline cardiomegaly', 'cardiac silhouette is enlarged', 'heart size is upper normal'],
    'pleural_effusion': ['pleural effusion', 'pleural effusions', 'effusion', 'effusions', 'pleural fluid'],
    'pneumothorax': ['pneumothorax', 'pneumothoraces', 'pneumothoraces'],
    'consolidation': ['consolidation', 'consolidations', 'consolidative'],
    'edema': ['edema', 'pulmonary edema', 'vascular congestion', 'pulmonary vascular congestion',
              'interstitial edema', 'congestive heart failure', 'chf'],
    'atelectasis': ['atelectasis', 'atelectatic', 'subsegmental atelectasis'],
    'pneumonia': ['pneumonia', 'pneumonitis', 'infection', 'infectious process'],
    'lung_opacity': ['opacity', 'opacities', 'infiltrate', 'infiltrates', 'airspace disease',
                     'air space disease', 'airspace opacity', 'air space opacity', 'airspace opacities',
                     'interstitial markings', 'interstitial opacities'],
    'nodule_mass': ['nodule', 'nodules', 'mass', 'masses', 'nodular opacity', 'lesion', 'lesions'],
    'emphysema_hyperinflation': ['emphysema', 'emphysematous', 'copd', 'hyperinflation', 'hyperinflated',
                                 'hyperexpansion', 'hyperexpanded', 'chronic obstructive pulmonary disease'],
    'granuloma_calcification': ['granuloma', 'granulomas', 'granulomatous', 'calcified granuloma',
                                'calcified granulomas', 'calcified nodule', 'calcified lymph nodes'],
    'scarring_fibrosis': ['scarring', 'scar', 'scars', 'fibrosis', 'fibrotic', 'fibrotic changes'],
    'fracture': ['fracture', 'fractures', 'fractured'],
    'pleural_thickening': ['pleural thickening', 'pleural scarring'],
    'hernia': ['hernia', 'hiatal hernia', 'hernias'],
    'degenerative_spine': ['degenerative changes', 'degenerative change', 'spondylosis', 'osteophytes',
                           'degenerative disease'],
    'support_device': ['pacemaker', 'catheter', 'tube', 'line', 'sternotomy wires', 'sternotomy',
                       'clips', 'surgical clips', 'port', 'stent', 'prosthesis', 'hardware', 'wires'],
    'aortic_tortuosity': ['tortuous', 'tortuosity', 'ectatic', 'aortic calcification',
                          'atherosclerotic', 'atherosclerosis', 'unfolded', 'unfolding'],
}

# "Normal" statements that assert absence of a finding without an explicit negation cue.
NORMAL_RULES = [
    # (required subject words, required predicate words, concept negated)
    ({'heart', 'cardiac', 'cardiomediastinal'}, {'normal', 'unremarkable', 'limits', 'stable'}, 'cardiomegaly'),
    ({'lung', 'lungs'}, {'clear', 'expanded'}, 'lung_opacity'),
    ({'osseous', 'bony', 'skeletal', 'bones'}, {'intact', 'normal', 'unremarkable'}, 'fracture'),
]

PRE_NEG = [['no'], ['not'], ['without'], ['negative', 'for'], ['free', 'of'], ['absence', 'of'],
           ['no', 'evidence', 'of'], ['no', 'definite'], ['no', 'visible'], ['no', 'acute'],
           ['clear', 'of'], ['resolution', 'of'], ['resolved']]
POST_NEG = [['is', 'not', 'seen'], ['are', 'not', 'seen'], ['not', 'seen'], ['not', 'identified'],
            ['not', 'present'], ['has', 'resolved'], ['have', 'resolved'], ['resolved'], ['absent'],
            ['are', 'absent'], ['is', 'absent'], ['are', 'negative'], ['is', 'negative']]
PRE_UNC = [['may'], ['might'], ['possible'], ['possibly'], ['probable'], ['probably'], ['likely'],
           ['questionable'], ['question', 'of'], ['suspicious', 'for'], ['suspicious'], ['suggest'],
           ['suggests'], ['suggestive', 'of'], ['suggesting'], ['concerning', 'for'], ['could'],
           ['cannot', 'exclude'], ['can', 'not', 'exclude'], ['cannot', 'be', 'excluded'],
           ['not', 'excluded'], ['versus'], ['vs'], ['borderline'], ['equivocal'], ['suspected'],
           ['differential'], ['would', 'be', 'difficult', 'to', 'exclude'], ['difficult', 'to', 'exclude'],
           ['upper', 'normal'], ['minimally']]
POST_UNC = [['cannot', 'be', 'excluded'], ['can', 'not', 'be', 'excluded'], ['not', 'excluded'],
            ['is', 'possible'], ['is', 'not', 'excluded'], ['not', 'entirely', 'excluded']]
# Scope terminators: a pre-cue before one of these does not reach findings after it.
TERMINATORS = {'but', 'however', 'although', 'though', 'otherwise', 'except', 'which', 'aside', 'apart',
               'and', 'with'}
# 'and'/'with' terminate *uncertainty* but NOT negation lists ("no effusion and no pneumothorax",
# "no effusion or pneumothorax").  Handled below.
NEG_LIST_CONTINUERS = {'or', 'and', 'nor', 'no', 'focal', 'pleural', 'acute', 'large', 'visible'}
MODIFIERS = {'left', 'right', 'bilateral', 'bibasilar', 'basilar', 'basal', 'upper', 'lower', 'mid', 'middle',
             'lobe', 'lobar', 'apical', 'apex', 'base', 'bases', 'lingula', 'lingular', 'perihilar', 'hilar',
             'retrocardiac', 'small', 'large', 'tiny', 'mild', 'mildly', 'moderate', 'moderately', 'severe',
             'minimal', 'trace', 'focal', 'patchy', 'streaky', 'diffuse', 'multifocal', 'calcified', 'new',
             'increased', 'decreased', 'chronic', 'acute', 'subsegmental', 'linear', 'interstitial',
             'pulmonary', 'pleural', 'airspace', 'stable', 'unchanged', 'old', 'healed', 'thoracic',
             'lumbar', 'spine', 'rib', 'ribs', 'lung', 'lungs', 'vertebral', 'endplate', 'costophrenic',
             'angle', 'angles', 'blunting', 'sided'}


def rules_sha256():
    src = inspect.getsource(sys.modules[__name__])
    return hashlib.sha256(src.encode('utf-8')).hexdigest()


def _patterns():
    pats = []
    for concept, phrases in CONCEPTS.items():
        for ph in phrases:
            pats.append((ph.split(), concept))
    pats.sort(key=lambda p: -len(p[0]))
    return pats


_PATTERNS = _patterns()


def _match_at(words, i, seq):
    return words[i:i + len(seq)] == seq


def split_sentences(words):
    """Return list of (start, end) half-open indices of sentences, excluding the '.' token."""
    out, s = [], 0
    for i, w in enumerate(words):
        if w == '.':
            if i > s:
                out.append((s, i))
            s = i + 1
    if s < len(words):
        out.append((s, len(words)))
    return out


def _find_findings(words, s, e):
    hits, i = [], s
    taken = [False] * len(words)
    while i < e:
        for seq, concept in _PATTERNS:
            if i + len(seq) <= e and _match_at(words, i, seq) and not any(taken[i:i + len(seq)]):
                hits.append((concept, i, i + len(seq)))
                for k in range(i, i + len(seq)):
                    taken[k] = True
                i += len(seq) - 1
                break
        i += 1
    return hits


def _last_cue_before(words, s, pos, cues, terminators):
    """Rightmost cue starting in [s, pos) with no terminator between cue end and pos."""
    best = None
    for i in range(s, pos):
        for cue in cues:
            if i + len(cue) <= pos and _match_at(words, i, cue):
                if not any(w in terminators for w in words[i + len(cue):pos]):
                    if best is None or i > best[0] or (i == best[0] and len(cue) > best[1] - best[0]):
                        best = (i, i + len(cue))
    return best


def _cue_after(words, pos, e, cues, max_gap=3):
    for i in range(pos, min(e, pos + max_gap + 1)):
        for cue in sorted(cues, key=len, reverse=True):
            if i + len(cue) <= e and _match_at(words, i, cue):
                return (i, i + len(cue))
    return None


def find_mentions(words):
    """List of dicts: concept, status, start, end (span incl. cue+modifiers), sentence."""
    mentions = []
    for s, e in split_sentences(words):
        sent = words[s:e]
        found = _find_findings(words, s, e)
        neg_term = TERMINATORS - {'and', 'with'}
        for concept, fs, fe in found:
            status, lo, hi = 'positive', fs, fe
            unc = _last_cue_before(words, s, fs, PRE_UNC, TERMINATORS)
            neg = _last_cue_before(words, s, fs, PRE_NEG, neg_term)
            post_unc = _cue_after(words, fe, e, POST_UNC)
            post_neg = _cue_after(words, fe, e, POST_NEG)
            # "no focal opacity to suggest pneumonia": a negation scoping over the uncertainty cue
            if unc is not None and neg is not None and neg[0] < unc[0]:
                unc = None
            if unc is not None or post_unc is not None:
                status = 'uncertain'
                if unc is not None:
                    lo = min(lo, unc[0])
                if post_unc is not None:
                    hi = max(hi, post_unc[1])
            elif neg is not None or post_neg is not None:
                status = 'negative'
                if neg is not None:
                    lo = min(lo, neg[0])
                if post_neg is not None:
                    hi = max(hi, post_neg[1])
            # expand over adjacent modifiers (anatomy / laterality / severity)
            while lo - 1 >= s and words[lo - 1] in MODIFIERS:
                lo -= 1
            while hi < e and words[hi] in MODIFIERS:
                hi += 1
            # "heart size is enlarged": whole clause; normal-size statements handled below
            if concept == 'cardiomegaly' and status == 'positive' and {'normal', 'unremarkable'} & set(sent):
                status = 'negative' if 'upper' not in sent else 'uncertain'
            mentions.append({'concept': concept, 'status': status, 'start': lo, 'end': hi, 'sentence': (s, e)})
        concepts_here = {m['concept'] for m in mentions if m['sentence'] == (s, e)}
        sw = set(sent)
        for subj, pred, concept in NORMAL_RULES:
            if concept in concepts_here:
                continue
            if sw & subj and sw & pred and not (sw & {'enlarged', 'enlargement', 'cardiomegaly'}):
                # whole short normality clause is the span ("the lungs are clear")
                st = 'negative'
                if concept == 'cardiomegaly' and sw & {'upper', 'borderline', 'top'}:
                    st = 'uncertain'  # "heart size is upper limits of normal"
                mentions.append({'concept': concept, 'status': st, 'start': s, 'end': e,
                                 'sentence': (s, e), 'normal_statement': True})
    return mentions


def report_labels(words):
    """concept -> status, aggregated per report. Priority: positive > uncertain > negative."""
    rank = {'negative': 0, 'uncertain': 1, 'positive': 2}
    out = {}
    for m in find_mentions(words):
        c, st = m['concept'], m['status']
        if c not in out or rank[st] > rank[out[c]]:
            out[c] = st
    return out


def token_weights(words, n_targets, alpha, drop_truncated=True):
    """Weights for the decoder targets (targets[1:]): position j <-> words[j]; position len(words) = EOS.
    Ordinary tokens (incl. '.', EOS) get 1, span tokens 1+alpha.  Spans cut by max_seq_length
    truncation are left at weight 1 when drop_truncated.  Returns (weights list, info dict)."""
    w = [1.0] * n_targets
    mentions = find_mentions(words)
    n_up, n_trunc = 0, 0
    for m in mentions:
        if m['end'] > n_targets and drop_truncated:
            n_trunc += 1
            continue
        for j in range(m['start'], min(m['end'], n_targets)):
            if w[j] == 1.0:
                n_up += 1
            w[j] = 1.0 + alpha
    return w, {'mentions': len(mentions), 'upweighted_tokens': n_up, 'truncated_spans': n_trunc,
               'n_targets': n_targets}


# ---------------------------------------------------------------- evaluation
def compare_labels(ref, gen):
    """Per-report error counts between two concept->status dicts."""
    pos = {'positive', 'uncertain'}
    r = {'tp_mention': 0, 'fp_mention': 0, 'fn_mention': 0,       # status-aware mention match
         'tp_abn': 0, 'fp_abn': 0, 'fn_abn': 0,                   # abnormal (pos/unc) findings
         'omissions': 0, 'polarity_errors': 0, 'uncertainty_errors': 0, 'unsupported_findings': 0}
    for c in set(ref) | set(gen):
        a, b = ref.get(c), gen.get(c)
        if a is not None and b is not None and a == b:
            r['tp_mention'] += 1
        else:
            if b is not None:
                r['fp_mention'] += 1
            if a is not None:
                r['fn_mention'] += 1
        ap, bp = a in pos, b in pos
        r['tp_abn'] += ap and bp
        r['fp_abn'] += bp and not ap
        r['fn_abn'] += ap and not bp
        if a is not None and b is None:
            r['omissions'] += 1
        if a is not None and b is not None:
            if {a, b} == {'positive', 'negative'}:
                r['polarity_errors'] += 1
            elif (a == 'uncertain') != (b == 'uncertain'):
                r['uncertainty_errors'] += 1
        if bp and not ap:
            r['unsupported_findings'] += 1
    return r


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float('nan')
    rc = tp / (tp + fn) if tp + fn else float('nan')
    f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float('nan')
    return p, rc, f


# ------------------------------------------------------------------ unit tests
POLARITY_TESTS = [
    ('there is no pleural effusion .', 'pleural_effusion', 'negative', 'no pleural effusion'),
    ('small left pleural effusion .', 'pleural_effusion', 'positive', 'small left pleural effusion'),
    ('cannot exclude edema .', 'edema', 'uncertain', 'cannot exclude edema'),
    ('edema cannot be excluded .', 'edema', 'uncertain', 'edema cannot be excluded'),
    ('no pneumothorax or pleural effusion .', 'pleural_effusion', 'negative', None),
    ('no pneumothorax or pleural effusion .', 'pneumothorax', 'negative', None),
    ('no focal consolidation pneumothorax or large pleural effusion .', 'pneumothorax', 'negative', None),
    ('possible right lower lobe pneumonia .', 'pneumonia', 'uncertain', 'possible right lower lobe pneumonia'),
    ('right lower lobe pneumonia .', 'pneumonia', 'positive', 'right lower lobe pneumonia'),
    ('pneumothorax is not seen .', 'pneumothorax', 'negative', 'pneumothorax is not seen'),
    ('the heart is normal in size .', 'cardiomegaly', 'negative', None),
    ('heart size is enlarged .', 'cardiomegaly', 'positive', None),
    ('the lungs are clear .', 'lung_opacity', 'negative', None),
    ('streaky bibasilar opacities likely atelectasis .', 'atelectasis', 'uncertain', None),
    ('streaky bibasilar opacities .', 'lung_opacity', 'positive', 'streaky bibasilar opacities'),
    ('no effusion but there is a small pneumothorax .', 'pneumothorax', 'positive', None),
    ('no effusion but there is a small pneumothorax .', 'pleural_effusion', 'negative', None),
    ('there is no focal air space opacity to suggest a pneumonia .', 'pneumonia', 'negative', None),
    ('the heart size is upper limits of normal .', 'cardiomegaly', 'uncertain', None),
    ('there are no nodules or masses .', 'nodule_mass', 'negative', None),
    ('no acute bony abnormality . degenerative changes of the spine .', 'degenerative_spine', 'positive', None),
]


def run_polarity_tests(verbose=True):
    failures = []
    for text, concept, status, span in POLARITY_TESTS:
        words = text.split()
        ms = [m for m in find_mentions(words) if m['concept'] == concept]
        got = report_labels(words).get(concept)
        ok = got == status
        if ok and span is not None:
            ok = any(' '.join(words[m['start']:m['end']]) == span for m in ms)
        if not ok:
            failures.append((text, concept, status, got, [' '.join(words[m['start']:m['end']]) for m in ms]))
    if verbose:
        print(f'polarity/scope tests: {len(POLARITY_TESTS) - len(failures)}/{len(POLARITY_TESTS)} passed')
        for f in failures:
            print('  FAIL', f)
    return failures


if __name__ == '__main__':
    sys.exit(1 if run_polarity_tests() else 0)
