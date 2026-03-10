# Op-Amp Autoresearch

Read `program.md` for full instructions. This is your bible.

## Quick Start
1. Read `program.md` completely
2. Read `specs.json` — target specifications (LM4562-class)
3. Read `design.cir` — current parametric netlist
4. Read `parameters.csv` — current DE sweep ranges
5. Run `python evaluate.py --quick` for fast baseline
6. Begin the experiment loop

## Key Commands
```bash
# Full DE optimization (adaptive budget — runs until convergence)
python evaluate.py 2>&1 | tee run.log

# Quick sanity check (small pop, low patience)
python evaluate.py --quick 2>&1 | tee run.log

# Remote evaluation (distributed workers)
python evaluate.py --server http://sim-node:8000 2>&1 | tee run.log
```

## Rules
- Modify ONLY `design.cir`, `parameters.csv`, and `evaluate.py`
- NEVER edit specs.json, program.md, or de/engine.py
- NEVER set parameter values yourself — define ranges, let DE optimize
- ALWAYS run evaluate.py after changing topology
- ALWAYS commit topology changes before running evaluation
- ALWAYS push results (including plots/) after evaluation
- NEVER stop iterating
