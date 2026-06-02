# casual_bayes

**A pocket calculator for "how worried should I actually be?"**

You just got a positive result on a scary medical test. An email looks like it might
be a scam. A smoke alarm went off. Your gut says one thing, but the *actual* odds are
often wildly different — and our intuition is famously bad at this.

`casual_bayes` does the math for you. You give it three plain-English numbers, and it
tells you the real updated probability, with notes explaining what just happened.

```
$ python3 casual_bayes.py classic --prior 1% --likelihood 90% --false-positive 9%

Starting belief / prior: 1.00%
Updated estimate / posterior: 9.17%
```

> A test that's "90% accurate" came back positive for a 1-in-100 disease — and there's
> still only a **9%** chance you actually have it. That's Bayes' theorem, and it's the
> kind of answer this tool exists to give you.

No dependencies, no setup, no internet. It's a single Python file.

---

## Quick start

```bash
# No install needed — just Python 3.8+
python3 casual_bayes.py classic --prior 1% --likelihood 90% --false-positive 9%
```

Run it with no arguments to see the built-in cheat sheet:

```bash
python3 casual_bayes.py
```

Want it as a command? Make it executable and drop it on your PATH:

```bash
chmod +x casual_bayes.py
mv casual_bayes.py ~/bin/casual_bayes   # or anywhere on your PATH
casual_bayes classic --prior 1% --likelihood 90% --false-positive 9%
```

---

## The only three numbers you need

Every everyday problem boils down to the same three questions. You can type them as
percentages (`20%`) or decimals (`0.2`) — whatever's easier.

| Term | What it means in plain English | Example |
|------|-------------------------------|---------|
| **prior** | Before any evidence, how common is this? The base rate. | "1% of people my age have this disease." |
| **likelihood** | If it's *true*, how often does this evidence show up? | "The test catches 90% of real cases." |
| **false-positive** | If it's *false*, how often does this evidence show up *anyway*? | "The test wrongly flags 9% of healthy people." |

The magic number people forget is the **false-positive rate**. A test can be "90%
accurate" and still be wrong most of the time when the thing it's testing for is rare.
That's the whole point of this tool.

---

## Which command do I want?

| Your situation | Command |
|----------------|---------|
| One piece of evidence, one yes/no question | `classic` |
| Same as classic, but you want to see the odds and the "Bayes factor" | `odds` |
| Several **independent** clues all pointing the same way | `naive` |
| Evidence arrives **over time**, and you update as you go | `sequential` |
| "What if I repeated this same test 3 times?" | `simulate` |
| You have **raw counts**, not percentages (e.g. 3 out of 50) | `counts` |
| You'd rather just describe the problem in a sentence | `extract` |

Add `--prompt` to any command to be asked for the numbers interactively, or `--json`
to get machine-readable output for scripts.

---

## Everyday examples

### "My test came back positive — should I panic?" (`classic`)

A disease affects **1%** of people. The test catches **90%** of real cases, but also
falsely flags **9%** of healthy people. You tested positive.

```bash
python3 casual_bayes.py classic --prior 1% --likelihood 90% --false-positive 9%
```

→ **Updated estimate: 9.17%.** A positive result raised your odds 9× — from 1% to ~9% —
but it's still far more likely to be a false alarm than the real thing. (This is exactly
why doctors retest before treating.)

### "Is this email a phishing scam?" (`naive`)

You can stack multiple independent clues. Say roughly **5%** of your incoming mail is
phishing (prior). This one has three red flags:

```bash
python3 casual_bayes.py naive --prior 5% --evidence '[
  {"name":"urgent wire-transfer ask","likelihood":0.7,"false_positive":0.02},
  {"name":"misspelled sender domain","likelihood":0.6,"false_positive":0.01},
  {"name":"link to a look-alike site","likelihood":0.8,"false_positive":0.03}
]'
```

→ Each clue multiplies the odds. Three weak-looking signals together push the estimate
to near-certainty. **Heads up:** `naive` assumes the clues are *independent*. If two of
your "clues" are really the same signal (e.g. a bad domain *and* a bad link are both just
"sketchy sender"), you'll overcount — don't double-dip.

### "Each new piece of news changes my mind" (`sequential`)

Use this when evidence shows up one step at a time and you want to watch your belief
move. The posterior from each round becomes the prior for the next.

```bash
python3 casual_bayes.py sequential --prior 10% --evidence '[
  {"name":"first lab result","likelihood":0.8,"false_positive":0.2},
  {"name":"confirmation test","likelihood":0.95,"false_positive":0.05}
]'
```

→ You get a round-by-round breakdown: `10% → 31% → 89%`, so you can see how much each
new result actually moved the needle.

### "How sure would I be if I just repeated the test?" (`simulate`)

```bash
python3 casual_bayes.py simulate --prior 1% --likelihood 90% --false-positive 9% --rounds 3
```

→ Shows what three identical positive results in a row would do to your estimate. Great
for "should I get a second (and third) opinion?" gut-checks. **Caveat:** real repeated
tests are rarely perfectly independent, so treat this as a sensitivity check, not gospel.

### "I have counts, not percentages" (`counts`)

If you're working from raw tallies (out of a study, a spreadsheet, your own records),
let the tool turn them into probabilities for you:

```bash
python3 casual_bayes.py counts --prompt
```

It'll ask for each count — how many had the condition, the totals, how often the
evidence appeared in each group — and then run a classic Bayes update.

### "Just let me describe it in words" (`extract`)

If you'd rather not figure out which number is which, describe the situation in plain
English and let Claude pull out the values. *(This one mode needs an API key — see below.)*

```bash
python3 casual_bayes.py extract "About 1% of people have this disease. The test detects \
it 90% of the time, but gives a false positive 9% of the time. I tested positive."
```

Add `--show-extracted` to see exactly what numbers it parsed before it does the math.

---

## Optional: natural-language mode setup

Only the `extract` command talks to the internet. Everything else runs fully offline.

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key"
# Optional — override the model:
export CASUAL_BAYES_HAIKU_MODEL="claude-haiku-4-5-20251001"
```

---

## Reading the output

Every result includes:

- **Prior → Posterior** — your belief before and after the evidence.
- **Bayes factor** — how many times more likely the evidence is when the hypothesis is
  true vs. false. Above 1 supports it; below 1 weakens it; `10x` means the evidence is
  10× more telling than a coin flip's worth.
- **Plain-English notes** — what the mode just did and what to watch out for.

Use `--json` on any command to get the raw numbers for piping into other tools.

---

## A word of caution

Bayes' theorem is only as honest as the numbers you feed it. The two classic traps:

1. **Forgetting the base rate.** Rare things stay rare even after a positive test.
2. **Double-counting evidence.** Two clues that are secretly the same signal will make
   you far too confident. When in doubt, use fewer, genuinely independent pieces of
   evidence.

This tool won't make those mistakes for you — but now you can see them coming.
