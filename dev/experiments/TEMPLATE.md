# <ID>: <short title>

- **Status:** planned | running | done
- **Owner:** <name>, <date>
- **Phase / task:** <e.g. 1.5>

## Question

One sentence. What decision does the answer change?

## Hypothesis

What we expect and why (cite the evidence: DESIGN.md finding, prior measurement).

## Change under test

The single difference between the arms. Arm A: <commit or option>. Arm B: <commit or
option>. Any switch used is registered in `dev/KNOBS.toml`.

## Method

- Cases: <catalogue and filter>
- Machines, compilers, instruction sets, thread modes:
- Metric: <e.g. median paired ratio B ÷ A, 95 % interval>
- Decision rule, written before running: <e.g. adopt B if the weighted ratio is below
  0.98 with the interval clear of the A/A band, and no case outside the targets is slower
  by more than 2 %>

## Results

Run IDs, the noise band from the A/A control, a short table. Link to the stored runs; do
not paste raw output.
When the change adds or removes code, the library's text size before and after
(`tools/size_report.py`, or the table in the pull request's job summaries).

## Verdict

Adopted / rejected / inconclusive, and what was done: default changed (commit), path
deleted (commit), DESIGN.md section updated, follow-up experiment opened, lesson added to
`dev/LESSONS.md` if the result generalises.
