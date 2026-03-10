#!/bin/bash
# Autonomous op-amp design loop.
# Runs Claude Code repeatedly — each iteration reads state, modifies topology,
# runs DE, commits, and pushes. If Claude exits, the loop restarts it.

cd "$(dirname "$0")"

ITERATION=0
LOG="run_loop.log"

echo "=== Op-Amp Autoresearch Loop Started: $(date) ===" | tee -a "$LOG"

while true; do
    ITERATION=$((ITERATION + 1))
    echo "" | tee -a "$LOG"
    echo "=== ITERATION $ITERATION — $(date) ===" | tee -a "$LOG"

    # The prompt tells Claude to do ONE full iteration of the experiment loop
    PROMPT="You are an autonomous analog circuit designer. Read program.md for full instructions.

Current iteration: $ITERATION

Do ONE complete experiment loop iteration:
1. Read results.tsv, design.cir, parameters.csv — understand current state
2. Analyze what's limiting performance (read measurements.json if it exists)
3. Plan and implement a topology change in design.cir and parameters.csv
4. Commit the topology change: git add design.cir parameters.csv evaluate.py && git commit -m 'topology: <description>'
5. Run: python3 evaluate.py 2>&1 | tee run.log
6. Read the evaluation output and analyze results
7. Append results to results.tsv
8. Commit and push everything: git add results.tsv plots/ measurements.json best_parameters.csv run.log && git commit -m 'results: <score> — <summary>' && git push

IMPORTANT: You MUST commit and push results before finishing. Do not skip any step."

    echo "$PROMPT" | claude --dangerously-skip-permissions 2>&1 | tee -a "$LOG"
    EXIT_CODE=$?

    echo "Claude exited with code $EXIT_CODE at $(date)" | tee -a "$LOG"

    # Brief pause between iterations
    sleep 5
done
