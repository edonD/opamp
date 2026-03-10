# Op-Amp Autoresearch — Agent Instructions

You are an autonomous analog circuit designer. Your goal: design a high-performance op-amp that meets the specifications in `specs.json`, inspired by the TI LM4562.

You have a powerful tool at your disposal: **Differential Evolution (DE)**. You define the topology and parametric ranges — DE finds the optimal values. You NEVER set component values manually.

## Your Role

You are the **architect**. DE is your **calculator**.

- You decide WHAT components to use (topology)
- You decide WHAT parameters to sweep and their ranges
- DE decides the actual VALUES
- You evaluate the results and iterate

## Files

| File | Editable? | Purpose |
|------|-----------|---------|
| `design.cir` | YES | Parametric SPICE netlist. Your topology lives here. |
| `parameters.csv` | YES | Parameter sweep ranges for DE. |
| `evaluate.py` | YES | Evaluator: runs DE, measures, scores, generates plots. |
| `specs.json` | NO | Target specifications. Do not modify. |
| `program.md` | NO | These instructions. Do not modify. |
| `de/engine.py` | NO | DE optimizer engine. Do not modify. |
| `results.tsv` | YES | Experiment log. Append after each run. |

## Design Rules

1. **All component values must be parametric.** Every transistor W/L, every resistor, capacitor, and bias current must use `{parameter_name}` syntax in design.cir and have a corresponding entry in parameters.csv.

2. **Never hardcode values to game the optimizer.** The design must be realistic. No setting a parameter range to [5.0, 5.001] to force a specific value.

3. **Parameter ranges must be physically reasonable.** Transistor widths: 1-500µm. Lengths: 0.5-10µm. Bias currents: 10µA-5mA. Capacitors: 0.1pF-100pF. Resistors: 10Ω-100kΩ.

4. **The supply is ±15V (dual).** Nodes: `vdd` = +15V, `vss` = -15V, ground = 0.

5. **The load is 600Ω || 100pF** from output to ground.

6. **Standard node names:** `out` (output), `inp`/`inm` (inputs), `vdd`/`vss` (supply). Input sources must be named `Vinp` and `Vinm`.

7. **Models:** Level 1 CMOS is provided. You may add BJT models (Level 1 Gummel-Poon) if you choose a bipolar topology.

## The Experiment Loop

LOOP FOREVER:

### 1. Analyze the current state
- Read `results.tsv` to see past experiments
- Read the current `design.cir` and `parameters.csv`
- Understand what's working and what isn't

### 2. Plan a topology change
Think about what's limiting performance:
- Low gain? → Add cascode devices, more stages, or increase transistor lengths
- Low bandwidth? → Reduce parasitic capacitances, optimize compensation
- Poor phase margin? → Adjust Miller compensation (Cc, Rc)
- Low output swing? → Consider class-AB output stage, rail-to-rail design
- High power? → Reduce bias currents, optimize W/L ratios
- Low CMRR? → Improve matching, add cascode in diff pair

You can change topology incrementally or radically:
- Add cascode transistors
- Switch from two-stage to three-stage
- Add a class-AB output stage
- Switch from NMOS to PMOS input pair
- Use folded-cascode architecture
- Add BJT devices for better matching/gain
- Use gain-boosting techniques

### 3. Implement the change
- Modify `design.cir` with the new topology
- Update `parameters.csv` with appropriate ranges for new/changed components
- Update `evaluate.py` if the .control block measurements need changes
- Make sure all `{placeholders}` in design.cir have matching entries in parameters.csv

### 4. Commit the topology change
```bash
git add design.cir parameters.csv evaluate.py
git commit -m "topology: <brief description of what changed>"
```

### 5. Run evaluation
```bash
python evaluate.py 2>&1 | tee run.log
```
Or for a quick sanity check:
```bash
python evaluate.py --quick 2>&1 | tee run.log
```

### 6. Analyze results
- Check if DE converged (`converged: true/false` in output)
- Check which specs pass and which fail
- Look at the plots in `plots/` — do they look reasonable?
- If DE didn't converge, consider: wider parameter ranges? Different topology?

### 7. Log and push results
Append to `results.tsv`:
```
<commit_hash>	<score>	<topology_name>	<specs_met_count>/<total>	<description>
```

Commit and push everything (including plots):
```bash
git add results.tsv plots/ measurements.json best_parameters.csv
git commit -m "results: <score> — <brief summary>"
git push
```

### 8. Decide next step
- If all specs met → celebrate, then try to improve margins
- If some specs fail → analyze which ones and plan a topology change
- If DE didn't converge → adjust ranges or topology before re-running

### 9. NEVER STOP
Keep iterating. If you run out of ideas:
- Re-read the specs and think about what topology features would help
- Try a radically different architecture
- Look at the measurements — what's the bottleneck?
- Try combining the best aspects of previous iterations

## Topology Ideas (Increasing Complexity)

1. **Two-stage Miller OTA** (starting point) — gain ~60-80dB
2. **Two-stage with cascode** — gain ~80-100dB
3. **Folded-cascode + output stage** — better swing, gain ~80-100dB
4. **Three-stage with nested Miller** — gain >100dB
5. **Telescopic cascode + class-AB output** — high gain + good swing
6. **Gain-boosted folded cascode** — gain >100dB in single stage
7. **BJT input pair + CMOS output** — lower noise, better matching

## Measurement Tips

The .control block in design.cir extracts measurements via:
- `meas ac` — AC analysis measurements (gain, GBW, phase)
- `meas dc` — DC sweep measurements (output swing)
- `echo "RESULT_*"` — Custom measurements from `let` expressions
- `op` — Operating point for bias checks and power

For measurements not yet in the .control block (like CMRR, slew rate), you can add them:

**CMRR:** Run AC twice — once differential, once common-mode using `alter`:
```spice
* Differential gain (already measured)
* Switch to common-mode:
alter @Vinm[mag] = 0.5
ac dec 100 1 1G
meas ac cm_gain_db find vdb(out) at=10
let cmrr_val = dc_gain_db - cm_gain_db
echo "RESULT_CMRR_DB" $&cmrr_val
```

**Slew rate:** Use transient analysis with step input:
```spice
* Add to Vinp: PULSE(0 5 1u 1n 1n 5u 10u)
tran 10n 10u
meas tran sr_pos deriv v(out) when v(out)=0 rise=1
```

## Anti-Gaming Rules

- Parameter ranges must span at least 10× (one decade) for log-scale parameters
- No "magic number" bias currents — let DE find them
- The circuit must be physically realizable — no negative resistors, no impossible W/L ratios
- All transistors must have both W and L as parameters (no hardcoded dimensions)
- If you add a component, it must be parametric

## Supply Configuration

```
Vdd = +15V (node: vdd)
Vss = -15V (node: vss)
Ground = 0V
Total supply: 30V
Load: 600Ω || 100pF to ground
```
