#!/usr/bin/env python3
"""
casual_bayes.py - single-file Bayesian estimation CLI.

Examples:
  python casual_bayes.py classic --prior 20% --likelihood 80% --false-positive 10%
  python casual_bayes.py classic --prompt
  python casual_bayes.py odds --prior 0.2 --likelihood 0.8 --false-positive 0.1
  python casual_bayes.py sequential --prior 0.2 --evidence '[{"name":"test A","likelihood":0.8,"false_positive":0.1}]'
  python casual_bayes.py naive --prior 0.2 --evidence '[{"name":"A","likelihood":0.8,"false_positive":0.1},{"name":"B","likelihood":0.7,"false_positive":0.2}]'
  python casual_bayes.py simulate --prior 0.2 --likelihood 0.8 --false-positive 0.1 --rounds 3
  python casual_bayes.py extract "Base rate is 20%, true positive is 80%, false positive is 10%."

Haiku extraction setup:
  pip install anthropic
  export ANTHROPIC_API_KEY="your-key"
  export CASUAL_BAYES_HAIKU_MODEL="claude-haiku-4-5-20251001"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Optional

DEFAULT_HAIKU_MODEL = "claude-haiku-4-5-20251001"
EPSILON = 1e-12


@dataclass
class Evidence:
    name: str
    likelihood: float
    false_positive: float


@dataclass
class Result:
    mode: str
    prior: float
    posterior: float
    likelihood: Optional[float] = None
    false_positive: Optional[float] = None
    bayes_factor: Optional[float] = None
    prior_odds: Optional[float] = None
    posterior_odds: Optional[float] = None
    evidence_probability: Optional[float] = None
    rounds: Optional[list[dict[str, Any]]] = None
    notes: Optional[list[str]] = None


def parse_probability(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing {name}.")
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("%"):
            x = float(raw[:-1].strip()) / 100.0
        else:
            x = float(raw)
    else:
        x = float(value)
    if 1.0 < x <= 100.0:
        x /= 100.0
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be 0..1 or 0%..100%; got {value!r}.")
    return x


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def num(x: Any) -> str:
    try:
        y = float(x)
        if math.isinf(y):
            return "infinite"
        return f"{y:.4g}"
    except Exception:
        return str(x)


def ask_probability(name: str) -> float:
    while True:
        raw = input(f"Enter {name} as a decimal or percent, e.g. 0.2 or 20%: ").strip()
        try:
            return parse_probability(raw, name)
        except ValueError as exc:
            print(f"  {exc}")


def get_probability(args: argparse.Namespace, attr: str, label: str) -> float:
    value = getattr(args, attr, None)
    if value is not None:
        return parse_probability(value, label)
    if getattr(args, "prompt", False):
        return ask_probability(label)
    raise ValueError(f"Missing --{attr.replace('_', '-')} or use --prompt.")


def safe_div(a: float, b: float, label: str) -> float:
    if abs(b) <= EPSILON:
        raise ValueError(f"Cannot divide by zero while computing {label}.")
    return a / b


def classic_bayes(prior: float, likelihood: float, false_positive: float) -> Result:
    numerator = prior * likelihood
    denominator = numerator + (1.0 - prior) * false_positive
    posterior = safe_div(numerator, denominator, "classic Bayes posterior")
    bayes_factor = math.inf if false_positive == 0 and likelihood > 0 else safe_div(likelihood, false_positive, "Bayes factor")
    return Result(
        mode="classic",
        prior=prior,
        likelihood=likelihood,
        false_positive=false_positive,
        posterior=posterior,
        evidence_probability=denominator,
        bayes_factor=bayes_factor,
        notes=[
            "Classic Bayes asks: after seeing this evidence, how much should belief in the hypothesis change?",
            "Likelihood means how often this evidence appears when the hypothesis is true.",
            "False-positive rate means how often this evidence still appears when the hypothesis is false.",
        ],
    )


def odds_bayes(prior: float, likelihood: float, false_positive: float) -> Result:
    if prior <= 0.0:
        return Result("odds", prior, 0.0, likelihood, false_positive, notes=["A 0% prior cannot be moved upward by finite evidence."])
    if prior >= 1.0:
        return Result("odds", prior, 1.0, likelihood, false_positive, notes=["A 100% prior cannot be moved downward by finite evidence."])
    prior_odds = prior / (1.0 - prior)
    bayes_factor = math.inf if false_positive == 0 and likelihood > 0 else safe_div(likelihood, false_positive, "Bayes factor")
    posterior_odds = prior_odds * bayes_factor
    posterior = posterior_odds / (1.0 + posterior_odds)
    return Result(
        mode="odds",
        prior=prior,
        likelihood=likelihood,
        false_positive=false_positive,
        posterior=posterior,
        bayes_factor=bayes_factor,
        prior_odds=prior_odds,
        posterior_odds=posterior_odds,
        notes=[
            "Odds form says: convert belief to odds, multiply by the Bayes factor, then convert back to probability.",
            "A Bayes factor above 1 supports the hypothesis; below 1 weakens it.",
        ],
    )


def sequential_bayes(prior: float, evidence: list[Evidence]) -> Result:
    current = prior
    rounds: list[dict[str, Any]] = []
    for i, ev in enumerate(evidence, 1):
        r = classic_bayes(current, ev.likelihood, ev.false_positive)
        rounds.append({
            "round": i,
            "name": ev.name,
            "prior_before": current,
            "likelihood": ev.likelihood,
            "false_positive": ev.false_positive,
            "posterior_after": r.posterior,
            "bayes_factor": r.bayes_factor,
        })
        current = r.posterior
    return Result(
        mode="sequential",
        prior=prior,
        posterior=current,
        rounds=rounds,
        notes=[
            "Sequential Bayes applies evidence one step at a time.",
            "The posterior from each round becomes the prior for the next round.",
        ],
    )


def naive_bayes(prior: float, evidence: list[Evidence]) -> Result:
    if not evidence:
        raise ValueError("Naive Bayes needs at least one evidence item.")
    if prior <= 0.0:
        return Result("naive", prior, 0.0, rounds=[], notes=["A 0% prior cannot be moved upward by finite evidence."])
    if prior >= 1.0:
        return Result("naive", prior, 1.0, rounds=[], notes=["A 100% prior cannot be moved downward by finite evidence."])

    prior_odds = prior / (1.0 - prior)
    total_bf = 1.0
    rows: list[dict[str, Any]] = []
    for i, ev in enumerate(evidence, 1):
        bf = math.inf if ev.false_positive == 0 and ev.likelihood > 0 else safe_div(ev.likelihood, ev.false_positive, f"Bayes factor for {ev.name}")
        total_bf *= bf
        rows.append({"item": i, "name": ev.name, "likelihood": ev.likelihood, "false_positive": ev.false_positive, "bayes_factor": bf})
    posterior_odds = prior_odds * total_bf
    posterior = posterior_odds / (1.0 + posterior_odds)
    return Result(
        mode="naive",
        prior=prior,
        posterior=posterior,
        bayes_factor=total_bf,
        prior_odds=prior_odds,
        posterior_odds=posterior_odds,
        rounds=rows,
        notes=[
            "Naive Bayes multiplies the Bayes factors from multiple evidence items.",
            "The key assumption is conditional independence; do not double-count evidence that is really the same signal twice.",
        ],
    )


def parse_evidence(raw: Optional[str]) -> list[Evidence]:
    if not raw:
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("--evidence must be a JSON list.")
    output = []
    for i, item in enumerate(data, 1):
        output.append(Evidence(
            name=str(item.get("name") or f"evidence {i}"),
            likelihood=parse_probability(item.get("likelihood"), f"evidence {i} likelihood"),
            false_positive=parse_probability(item.get("false_positive"), f"evidence {i} false_positive"),
        ))
    return output


def ask_evidence() -> list[Evidence]:
    print("Add evidence items. Leave name blank when done.")
    items: list[Evidence] = []
    while True:
        name = input("Evidence name: ").strip()
        if not name:
            break
        items.append(Evidence(
            name=name,
            likelihood=ask_probability(f"{name} likelihood P(E|H)"),
            false_positive=ask_probability(f"{name} false-positive P(E|not H)"),
        ))
    return items


def extract_values_with_haiku(text: str) -> dict[str, Any]:
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Haiku extraction requires the Anthropic SDK: pip install anthropic") from exc

    client = anthropic.Anthropic()
    model = os.getenv("CASUAL_BAYES_HAIKU_MODEL", os.getenv("CAUSAL_BAYES_HAIKU_MODEL", DEFAULT_HAIKU_MODEL))
    prompt = f"""
Extract Bayesian estimation values from the user's situation.
Return ONLY valid JSON, no markdown.

Schema:
{{
  "situation": "plain English description",
  "prior": number_or_null,
  "likelihood": number_or_null,
  "false_positive": number_or_null,
  "evidence": [{{"name": "short name", "likelihood": number, "false_positive": number}}],
  "recommended_mode": "classic|odds|sequential|naive"
}}

Rules:
- Convert percentages to decimals: 20% -> 0.2.
- Do not invent numbers; use null when missing.
- Multiple independent evidence items imply naive mode.
- Ordered evidence rounds imply sequential mode.
- One evidence item implies classic mode.

User situation:
{text}
""".strip()
    response = client.messages.create(
        model=model,
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "\n".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f"Haiku did not return JSON. Raw response:\n{raw}")
        return json.loads(match.group(0))


def explain(result: Result, situation: Optional[str] = None) -> str:
    lines: list[str] = []
    if situation:
        lines += [f"Situation: {situation}", ""]
    lines += [f"Mode: {result.mode}", f"Starting belief / prior: {pct(result.prior)}", f"Updated estimate / posterior: {pct(result.posterior)}", ""]
    if result.likelihood is not None:
        lines += [
            f"Likelihood P(evidence | hypothesis true): {pct(result.likelihood)}",
            f"Natural language: if the hypothesis is true, this evidence shows up about {pct(result.likelihood)} of the time.",
        ]
    if result.false_positive is not None:
        lines += [
            f"False-positive rate P(evidence | hypothesis false): {pct(result.false_positive)}",
            f"Natural language: even if the hypothesis is false, this evidence still shows up about {pct(result.false_positive)} of the time.",
        ]
    if result.bayes_factor is not None:
        lines.append(f"Bayes factor: {num(result.bayes_factor)}x")
        if math.isfinite(float(result.bayes_factor)) and result.bayes_factor > 1:
            lines.append(f"Natural language: the evidence is {num(result.bayes_factor)} times more common when the hypothesis is true than when it is false.")
        elif math.isfinite(float(result.bayes_factor)) and result.bayes_factor < 1:
            lines.append(f"Natural language: the evidence is {num(1 / result.bayes_factor)} times less common when the hypothesis is true than when it is false.")
    if result.prior_odds is not None:
        lines.append(f"Prior odds: {num(result.prior_odds)}")
    if result.posterior_odds is not None:
        lines.append(f"Posterior odds: {num(result.posterior_odds)}")
    if result.evidence_probability is not None:
        lines.append(f"Overall probability of seeing this evidence under the model: {pct(result.evidence_probability)}")
    if result.rounds:
        lines += ["", "Evidence/update breakdown:"]
        for row in result.rounds:
            if "posterior_after" in row:
                lines.append(f"  Round {row['round']} ({row['name']}): {pct(row['prior_before'])} -> {pct(row['posterior_after'])}; BF={num(row['bayes_factor'])}")
            else:
                lines.append(f"  {row['name']}: likelihood={pct(row['likelihood'])}, false-positive={pct(row['false_positive'])}, BF={num(row['bayes_factor'])}")
    if result.notes:
        lines += ["", "Plain-English notes:"] + [f"  - {note}" for note in result.notes]
    lines += ["", "Caution: the estimate is only as good as the assumptions. Correlated or double-counted evidence can make the result too confident."]
    return "\n".join(lines)


def print_result(result: Result, args: argparse.Namespace, situation: Optional[str] = None) -> None:
    if getattr(args, "json", False):
        payload = asdict(result)
        if situation:
            payload["situation"] = situation
        print(json.dumps(payload, indent=2))
    else:
        print(explain(result, situation))


def handle_classic(args: argparse.Namespace) -> None:
    print_result(classic_bayes(
        get_probability(args, "prior", "prior"),
        get_probability(args, "likelihood", "likelihood"),
        get_probability(args, "false_positive", "false-positive rate"),
    ), args)


def handle_odds(args: argparse.Namespace) -> None:
    print_result(odds_bayes(
        get_probability(args, "prior", "prior"),
        get_probability(args, "likelihood", "likelihood"),
        get_probability(args, "false_positive", "false-positive rate"),
    ), args)


def handle_sequential(args: argparse.Namespace) -> None:
    prior = get_probability(args, "prior", "prior")
    evidence = ask_evidence() if args.prompt else parse_evidence(args.evidence)
    if not evidence:
        raise ValueError("Sequential mode needs --evidence or --prompt.")
    print_result(sequential_bayes(prior, evidence), args)


def handle_naive(args: argparse.Namespace) -> None:
    prior = get_probability(args, "prior", "prior")
    evidence = ask_evidence() if args.prompt else parse_evidence(args.evidence)
    if not evidence:
        raise ValueError("Naive mode needs --evidence or --prompt.")
    print_result(naive_bayes(prior, evidence), args)


def handle_simulate(args: argparse.Namespace) -> None:
    prior = get_probability(args, "prior", "prior")
    likelihood = get_probability(args, "likelihood", "likelihood")
    false_positive = get_probability(args, "false_positive", "false-positive rate")
    rounds = max(1, int(args.rounds))
    evidence = [Evidence(f"same evidence repeated {i}", likelihood, false_positive) for i in range(1, rounds + 1)]
    result = sequential_bayes(prior, evidence)
    result.mode = "simulation"
    result.notes = [
        "Simulation here means repeatedly applying the same Bayesian update.",
        "This is a sensitivity check; repeated identical evidence is often not independent in real life.",
    ]
    print_result(result, args)


def handle_counts(args: argparse.Namespace) -> None:
    def get_count(attr: str) -> float:
        val = getattr(args, attr)
        if val is not None:
            x = float(val)
        elif args.prompt:
            x = float(input(f"Enter {attr.replace('_', ' ')}: "))
        else:
            raise ValueError(f"Missing --{attr.replace('_', '-')}, or use --prompt.")
        if x < 0:
            raise ValueError(f"{attr} must be non-negative.")
        return x
    h = get_count("hypothesis_count")
    total = get_count("total_count")
    e_h = get_count("evidence_given_h_count")
    h_denom = get_count("h_count_for_likelihood")
    e_not_h = get_count("evidence_given_not_h_count")
    not_h_denom = get_count("not_h_count_for_false_positive")
    result = classic_bayes(
        safe_div(h, total, "prior from counts"),
        safe_div(e_h, h_denom, "likelihood from counts"),
        safe_div(e_not_h, not_h_denom, "false-positive rate from counts"),
    )
    result.mode = "counts -> classic"
    result.notes = [
        "The tool first converted your counts into prior, likelihood, and false-positive rate.",
        f"Prior = {h:g} / {total:g}.",
        f"Likelihood = {e_h:g} / {h_denom:g}.",
        f"False-positive rate = {e_not_h:g} / {not_h_denom:g}.",
    ]
    print_result(result, args)


def handle_extract(args: argparse.Namespace) -> None:
    extracted = extract_values_with_haiku(args.text)
    situation = extracted.get("situation")
    prior = extracted.get("prior")
    if prior is None:
        prior = ask_probability("prior")
    prior = parse_probability(prior, "prior")
    evidence = []
    for i, item in enumerate(extracted.get("evidence") or [], 1):
        if item.get("likelihood") is not None and item.get("false_positive") is not None:
            evidence.append(Evidence(str(item.get("name") or f"evidence {i}"), parse_probability(item["likelihood"], "likelihood"), parse_probability(item["false_positive"], "false-positive")))
    mode = extracted.get("recommended_mode") or "classic"
    if len(evidence) > 1 and mode == "naive":
        result = naive_bayes(prior, evidence)
    elif len(evidence) > 1 and mode == "sequential":
        result = sequential_bayes(prior, evidence)
    else:
        likelihood = extracted.get("likelihood", evidence[0].likelihood if evidence else None)
        false_positive = extracted.get("false_positive", evidence[0].false_positive if evidence else None)
        if likelihood is None:
            likelihood = ask_probability("likelihood")
        if false_positive is None:
            false_positive = ask_probability("false-positive rate")
        result = classic_bayes(prior, parse_probability(likelihood, "likelihood"), parse_probability(false_positive, "false-positive"))
    if args.show_extracted:
        print("Extracted values:")
        print(json.dumps(extracted, indent=2))
        print()
    print_result(result, args, situation)



RUNTIME_README = """
casual_bayes — friendly Bayesian estimation from the command line

Examples:
  python casual_bayes.py classic --prior 20% --likelihood 80% --false-positive 10%
  python casual_bayes.py classic --prompt
  python casual_bayes.py odds --prior 0.2 --likelihood 0.8 --false-positive 0.1
  python casual_bayes.py sequential --prior 0.2 --evidence '[{"name":"test A","likelihood":0.8,"false_positive":0.1}]'
  python casual_bayes.py naive --prior 0.2 --evidence '[{"name":"A","likelihood":0.8,"false_positive":0.1},{"name":"B","likelihood":0.7,"false_positive":0.2}]'
  python casual_bayes.py simulate --prior 0.2 --likelihood 0.8 --false-positive 0.1 --rounds 3
  python casual_bayes.py counts --prompt

Haiku extraction setup:
  pip install anthropic
  export ANTHROPIC_API_KEY="your-key"
  export CASUAL_BAYES_HAIKU_MODEL="claude-haiku-4-5-20251001"  # optional

Haiku extraction example:
  python casual_bayes.py extract "A company has two machines. Machine A makes 70% of the products, and Machine B makes 30%. Machine A has a defect rate of 2%, while Machine B has a defect rate of 6%. A randomly selected product is defective. What is the probability it came from Machine B?."

Modes:
  classic     Binary Bayes using prior, likelihood, and false-positive rate.
  odds        Odds-form Bayes using prior odds multiplied by a Bayes factor.
  sequential  Repeated updates where each posterior becomes the next prior.
  naive       Multiple conditionally independent evidence items.
  simulate    Repeatedly applies the same update as a sensitivity check.
  counts      Converts count data into probabilities, then runs classic Bayes.
  extract     Uses Claude Haiku to pull values out of natural language.
""".strip()

def add_basic(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prior")
    p.add_argument("--likelihood")
    p.add_argument("--false-positive", dest="false_positive")
    p.add_argument("--prompt", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="casual_bayes",
        description="Friendly Bayesian estimation CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=RUNTIME_README,
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of natural language.")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("classic", help="Classic binary Bayes."); add_basic(p); p.set_defaults(func=handle_classic)
    p = sub.add_parser("odds", help="Odds-form Bayes."); add_basic(p); p.set_defaults(func=handle_odds)

    p = sub.add_parser("sequential", help="Sequential updates from evidence JSON or prompts.")
    p.add_argument("--prior"); p.add_argument("--evidence"); p.add_argument("--prompt", action="store_true"); p.set_defaults(func=handle_sequential)

    p = sub.add_parser("naive", help="Naive Bayes over independent evidence items.")
    p.add_argument("--prior"); p.add_argument("--evidence"); p.add_argument("--prompt", action="store_true"); p.set_defaults(func=handle_naive)

    p = sub.add_parser("simulate", help="Repeat the same update for N rounds."); add_basic(p); p.add_argument("--rounds", type=int, default=3); p.set_defaults(func=handle_simulate)

    p = sub.add_parser("counts", help="Convert count data into classic Bayes inputs.")
    for attr in ["hypothesis_count", "total_count", "evidence_given_h_count", "h_count_for_likelihood", "evidence_given_not_h_count", "not_h_count_for_false_positive"]:
        p.add_argument("--" + attr.replace("_", "-"), dest=attr)
    p.add_argument("--prompt", action="store_true"); p.set_defaults(func=handle_counts)

    p = sub.add_parser("extract", help="Use Claude Haiku to extract values from natural language.")
    p.add_argument("text"); p.add_argument("--show-extracted", action="store_true"); p.set_defaults(func=handle_extract)
    return parser


def main() -> int:
    if len(sys.argv) == 1:
        print(RUNTIME_README)
        print("\nRun `python casual_bayes.py -h` for full command help.")
        return 0

    parser = build_parser()
    args = parser.parse_args()
    try:
        if not hasattr(args, "func"):
            print(RUNTIME_README)
            print("\nRun `python casual_bayes.py -h` for full command help.")
            return 0
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
