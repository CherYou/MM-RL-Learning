"""Deterministic learning verifiers; benchmark scope is documented per dataset."""

from collections import Counter
from decimal import Decimal, InvalidOperation
import re
import string


def extract_answer(text):
    tags = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if tags:
        return tags[-1].strip()
    # Balanced braces support nested boxed fractions without eval/sympify.
    start = text.rfind("\\boxed{")
    if start >= 0:
        depth = 1
        for end in range(start + 7, len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if depth == 0:
                return text[start + 7 : end].strip()
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    match = re.search(r"(?:Answer|answer)\s*:\s*(.+)", text)
    return match.group(1).strip() if match else text.strip()


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def mathematical_equivalence(pred, gold):
    """Bounded symbolic comparison; free-form prose never becomes a numeric match."""

    def mathematical(text):
        return len(text) <= 500 and ("\\" in text or re.fullmatch(r"[\d\s.,+*/^(){}\[\]%=<>xyzeiπ−-]+", text))

    if not mathematical(pred) or not mathematical(gold):
        return False
    from math_verify import parse, verify

    try:
        target = parse("$" + pred.strip("$") + "$", parsing_timeout=1)
        reference = parse("$" + gold.strip("$") + "$", parsing_timeout=1)
        return bool(target and reference and verify(reference, target, timeout_seconds=1))
    except Exception:
        return False


def exact_match(prediction, answers):
    if isinstance(answers, str):
        answers = [answers]
    pred = extract_answer(prediction)
    for gold in answers:
        if pred.strip() == str(gold).strip():
            return 1.0
        try:
            if Decimal(pred.replace(",", "")) == Decimal(str(gold).replace(",", "")):
                return 1.0
        except InvalidOperation:
            pass
        if normalize(pred) and normalize(pred) == normalize(str(gold)):
            return 1.0
        if mathematical_equivalence(pred, str(gold)):
            return 1.0
    return 0.0


def token_f1(prediction, gold):
    p = normalize(extract_answer(prediction)).split()
    g = normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    return 2 * common / (len(p) + len(g)) if p and g else float(p == g)


def reward(sample, row, kind="math"):
    if kind == "debug_token":
        # Explicit smoke reward, unrelated to task correctness; never used in experiment configs.
        return sum((token % 7 == 0) * keep for token, keep in zip(sample.tokens, sample.mask))
    if kind == "f1":
        return max(token_f1(sample.text, str(a)) for a in row.get("answers", [row["answer"]]))
    return exact_match(sample.text, row.get("answers") or row["answer"])
