"""
TELUGU/PHASE3/reasoning_data/generate_telugu_reasoning.py

Builds the synthetic comparative reasoning dataset for Model H (Telugu):
pairwise, threeway, transitive templates, plus a transitive_4hop pattern kept
only in test to check generalization. Names are split into disjoint
train/val/test pools so nothing leaks across splits.
"""

import json
import random
from pathlib import Path

SEED = 1027
OUT_DIR = Path(__file__).resolve().parent

PERSON_NAMES = [
    "రాము", "సీత", "గీత", "రవి", "కిరణ్", "అనిత", "సురేష్", "లక్ష్మి", "మోహన్", "ప్రియ",
    "విజయ్", "దీప", "అర్జున్", "మీర", "కృష్ణ", "రాధ", "గోపాల్", "సునీత", "రమేష్", "పద్మ",
    "హరి", "శ్రీను", "జయ", "వేణు", "స్వాతి", "రాజు", "కావ్య", "బాబు", "నందిని", "శంకర్",
    "పార్వతి", "అనిల్", "సంధ్య", "నరేష్", "వాణి", "ప్రసాద్", "శ్రీదేవి", "మురళి", "ఉష", "చంద్ర",
    "వెంకటేష్", "సాయి", "దుర్గ", "రామకృష్ణ", "సుజాత", "నాగరాజు", "విద్య", "శ్రీనివాస్", "అరుణ", "గణేష్",
    "భవాని", "రాజేష్", "కళ్యాణి", "మురళీధర్", "సుమతి", "యోగేష్", "చంద్రిక", "ప్రభాకర్", "రేఖ", "సతీష్",
    "అనూష", "విక్రమ్", "సౌమ్య", "రాఘవ", "పల్లవి", "శేఖర్", "మంజుల", "గోవింద్", "రమణ", "సింధు",
]

OBJECT_NAMES = [
    "పుస్తకం", "సైకిల్", "ఫోన్", "గడియారం", "బ్యాగ్", "కుర్చీ", "టేబుల్", "పెన్ను", "బ్యాట్", "కారు",
    "గొడుగు", "బూట్లు", "ల్యాప్‌టాప్", "టీవీ", "ఫ్యాన్", "వాచ్", "సూట్‌కేస్", "కెమెరా", "హెడ్‌ఫోన్", "బల్ల",
    "సోఫా", "బెడ్", "అద్దం", "బకెట్", "కప్పు", "ప్లేటు", "బొమ్మ", "బుక్‌షెల్ఫ్", "వాషింగ్ మెషిన్", "ఫ్రిజ్",
    "మిక్సీ", "స్కూటర్",
]

ATTRIBUTES = {
    "age": {"noun": "వయస్సు", "unit": "సంవత్సరాలు", "lo": 5, "hi": 90, "uses": "person",
            "more_q": "వీరిలో ఎవరు వయసులో పెద్దవారు?", "less_q": "వీరిలో ఎవరు వయసులో చిన్నవారు?"},
    "height": {"noun": "ఎత్తు", "unit": "సెం.మీ", "lo": 90, "hi": 200, "uses": "person",
               "more_q": "వీరిలో ఎవరు ఎత్తులో పొడవుగా ఉన్నారు?", "less_q": "వీరిలో ఎవరు ఎత్తులో పొట్టిగా ఉన్నారు?"},
    "price": {"noun": "ధర", "unit": "రూపాయలు", "lo": 50, "hi": 50000, "uses": "object",
              "more_q": "వీటిలో ఏది ఖరీదైనది?", "less_q": "వీటిలో ఏది చౌకైనది?"},
    "weight": {"noun": "బరువు", "unit": "కిలోలు", "lo": 1, "hi": 100, "uses": "object",
               "more_q": "వీటిలో ఏది బరువైనది?", "less_q": "వీటిలో ఏది తేలికైనది?"},
}

ANSWER_PREFIX = "\nసమాధానం:"

STATEMENT_VARIANTS = [
    "{name} {noun} {val} {unit}.",
    "{name} యొక్క {noun} {val} {unit}.",
    "{name}కి {val} {unit} {noun} ఉంది.",
]
GENERIC_MORE_Q = [
    "వీరిలో ఎవరు {noun}లో ఎక్కువ ఉన్నారు?",
    "వీరిద్దరిలో ఎవరికి ఎక్కువ {noun} ఉంది?",
    "{noun} పరంగా వీరిలో ఎవరు ఎక్కువ?",
]
GENERIC_LESS_Q = [
    "వీరిలో ఎవరు {noun}లో తక్కువ ఉన్నారు?",
    "వీరిద్దరిలో ఎవరికి తక్కువ {noun} ఉంది?",
    "{noun} పరంగా వీరిలో ఎవరు తక్కువ?",
]
GENERIC_MOST_Q = [
    "వీరిలో అందరికంటే ఎక్కువ {noun} ఉన్నవారు ఎవరు?",
    "వీరి ముగ్గురిలో ఎక్కువ {noun} ఉన్నవారు ఎవరు?",
    "{noun} పరంగా వీరిలో అందరికంటే ఎక్కువ ఎవరు?",
]
GENERIC_LEAST_Q = [
    "వీరిలో అందరికంటే తక్కువ {noun} ఉన్నవారు ఎవరు?",
    "వీరి ముగ్గురిలో తక్కువ {noun} ఉన్నవారు ఎవరు?",
    "{noun} పరంగా వీరిలో అందరికంటే తక్కువ ఎవరు?",
]
RELATION_MORE_VARIANTS = [
    "{lo} కంటే {hi} {noun}లో ఎక్కువ.",
    "{hi} {noun}, {lo} కంటే ఎక్కువ.",
    "{lo}తో పోలిస్తే {hi}కి ఎక్కువ {noun} ఉంది.",
]


def split_pool(items, val_frac=0.15, test_frac=0.2, seed=0):
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_val, n_test = max(1, int(n * val_frac)), max(1, int(n * test_frac))
    return items[n_val + n_test:], items[:n_val], items[n_val:n_val + n_test]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def dedup_examples(examples):
    seen, out = set(), []
    for ex in examples:
        if ex["prompt"] in seen:
            continue
        seen.add(ex["prompt"])
        out.append(ex)
    return out


def entity_pool(attr_key):
    return "person" if ATTRIBUTES[attr_key]["uses"] == "person" else "object"


def sample_distinct_values(rng, lo, hi, n):
    vals = set()
    while len(vals) < n:
        vals.add(rng.randint(lo, hi))
    return list(vals)


def render_statement(rng, name, noun, val, unit):
    return rng.choice(STATEMENT_VARIANTS).format(name=name, noun=noun, val=val, unit=unit)


def make_pairwise(rng, names, attr_key):
    attr = ATTRIBUTES[attr_key]
    a, b = rng.sample(names, 2)
    va, vb = sample_distinct_values(rng, attr["lo"], attr["hi"], 2)
    ask_more = rng.random() < 0.5
    question = rng.choice([attr["more_q"]] + [t.format(noun=attr["noun"]) for t in GENERIC_MORE_Q]) if ask_more \
        else rng.choice([attr["less_q"]] + [t.format(noun=attr["noun"]) for t in GENERIC_LESS_Q])
    winner = a if (va > vb) == ask_more else b
    s1 = render_statement(rng, a, attr["noun"], va, attr["unit"])
    s2 = render_statement(rng, b, attr["noun"], vb, attr["unit"])
    return {"prompt": f"{s1} {s2} {question}{ANSWER_PREFIX}", "answer": f" {winner}", "template": "pairwise", "attribute": attr_key}


def make_threeway(rng, names, attr_key):
    attr = ATTRIBUTES[attr_key]
    a, b, c = rng.sample(names, 3)
    va, vb, vc = sample_distinct_values(rng, attr["lo"], attr["hi"], 3)
    ask_most = rng.random() < 0.5
    pairs = [(a, va), (b, vb), (c, vc)]
    winner = max(pairs, key=lambda p: p[1])[0] if ask_most else min(pairs, key=lambda p: p[1])[0]
    q = rng.choice(GENERIC_MOST_Q if ask_most else GENERIC_LEAST_Q).format(noun=attr["noun"])
    s1 = render_statement(rng, a, attr["noun"], va, attr["unit"])
    s2 = render_statement(rng, b, attr["noun"], vb, attr["unit"])
    s3 = render_statement(rng, c, attr["noun"], vc, attr["unit"])
    return {"prompt": f"{s1} {s2} {s3} {q}{ANSWER_PREFIX}", "answer": f" {winner}", "template": "threeway", "attribute": attr_key}


def make_transitive(rng, names, attr_key):
    attr = ATTRIBUTES[attr_key]
    x1, x2, x3 = rng.sample(names, 3)
    ask_more = rng.random() < 0.5
    tail_q = "ఎక్కువ" if ask_more else "తక్కువ"
    winner = x3 if ask_more else x1
    r1 = rng.choice(RELATION_MORE_VARIANTS).format(lo=x1, hi=x2, noun=attr["noun"])
    r2 = rng.choice(RELATION_MORE_VARIANTS).format(lo=x2, hi=x3, noun=attr["noun"])
    prompt = f"{r1} {r2} {x1} మరియు {x3} లో ఎవరికి {tail_q} {attr['noun']} ఉంది?{ANSWER_PREFIX}"
    return {"prompt": prompt, "answer": f" {winner}", "template": "transitive", "attribute": attr_key}


def make_transitive_4hop(rng, names, attr_key):
    """Held-out relation pattern: 3-hop chain over 4 entities, test-only."""
    attr = ATTRIBUTES[attr_key]
    x1, x2, x3, x4 = rng.sample(names, 4)
    ask_more = rng.random() < 0.5
    tail_q = "ఎక్కువ" if ask_more else "తక్కువ"
    winner = x4 if ask_more else x1
    r1 = rng.choice(RELATION_MORE_VARIANTS).format(lo=x1, hi=x2, noun=attr["noun"])
    r2 = rng.choice(RELATION_MORE_VARIANTS).format(lo=x2, hi=x3, noun=attr["noun"])
    r3 = rng.choice(RELATION_MORE_VARIANTS).format(lo=x3, hi=x4, noun=attr["noun"])
    prompt = f"{r1} {r2} {r3} {x1} మరియు {x4} లో ఎవరికి {tail_q} {attr['noun']} ఉంది?{ANSWER_PREFIX}"
    return {"prompt": prompt, "answer": f" {winner}", "template": "transitive_4hop", "attribute": attr_key}


def generate_split(rng, person_names, object_names, n_per_template, template_fns):
    examples = []
    for _, (fn, min_entities) in template_fns.items():
        count, attempts = 0, 0
        while count < n_per_template and attempts < n_per_template * 40:
            attempts += 1
            attr_key = rng.choice(list(ATTRIBUTES.keys()))
            names = person_names if entity_pool(attr_key) == "person" else object_names
            if len(names) < min_entities:
                continue
            examples.append(fn(rng, names, attr_key))
            count += 1
    return dedup_examples(examples)


def write_dataset_stats(train, val, test):
    templates = sorted({r["template"] for r in train + val + test})
    lines = ["# Telugu (Model H) — Synthetic Reasoning Dataset Statistics\n", "## Split sizes\n",
              "| Split | Examples |", "|---|---|",
              f"| train | {len(train)} |", f"| val | {len(val)} |", f"| test | {len(test)} |",
              "\n## Template counts per split\n", "| Split | " + " | ".join(templates) + " |",
              "|---|" + "---|" * len(templates)]
    for name, recs in [("train", train), ("val", val), ("test", test)]:
        counts = [str(sum(1 for r in recs if r["template"] == t)) for t in templates]
        lines.append(f"| {name} | " + " | ".join(counts) + " |")
    lines.append(
        "\n## Leakage avoidance\n\nPerson/object names are split into disjoint train/val/test "
        "pools before any example is generated. `transitive_4hop` (a 3-hop chain over 4 "
        "entities) is generated only for test -- a held-out relation pattern never seen "
        "during finetuning. Every template carries multiple statement/question phrasing "
        "variants (randomly chosen per example) to avoid one-shape overfitting."
    )
    (OUT_DIR / "dataset_stats.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    rng = random.Random(SEED)
    person_train, person_val, person_test = split_pool(PERSON_NAMES, seed=SEED)
    object_train, object_val, object_test = split_pool(OBJECT_NAMES, seed=SEED + 1)

    core = {"pairwise": (make_pairwise, 2), "threeway": (make_threeway, 3), "transitive": (make_transitive, 3)}
    train = generate_split(rng, person_train, object_train, 1500, core)
    val = generate_split(rng, person_val, object_val, 250, core)
    test_core = generate_split(rng, person_test, object_test, 250, core)
    test_gen = generate_split(rng, person_test, object_test, 400, {"transitive_4hop": (make_transitive_4hop, 4)})
    test = dedup_examples(test_core + test_gen)

    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)
    write_jsonl(OUT_DIR / "test.jsonl", test)
    write_dataset_stats(train, val, test)
    print(f"[generate_telugu_reasoning] train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
