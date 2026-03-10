"""
evaluate.py — Evaluator for op-amp autoresearch.

Reads design.cir + parameters.csv + specs.json, runs DE optimization,
extracts ngspice measurements, scores against specs, generates plots.

Usage:
    python evaluate.py                          # local (uses all CPU cores)
    python evaluate.py --server http://host:8000 # remote sim server
    python evaluate.py --quick                   # fast check (small pop, few iters)
"""

import os
import sys
import re
import json
import csv
import time
import argparse
import subprocess
import tempfile
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NGSPICE = os.environ.get("NGSPICE", "/usr/local/bin/ngspice")
DESIGN_FILE = "design.cir"
PARAMS_FILE = "parameters.csv"
SPECS_FILE = "specs.json"
RESULTS_FILE = "results.tsv"
PLOTS_DIR = "plots"

# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------

def load_parameters(path: str = PARAMS_FILE) -> List[Dict]:
    """Load parameter definitions from CSV."""
    params = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            params.append({
                "name": row["name"].strip(),
                "min": float(row["min"]),
                "max": float(row["max"]),
                "scale": row.get("scale", "lin").strip(),
            })
    return params


def load_design(path: str = DESIGN_FILE) -> str:
    with open(path) as f:
        return f.read()


def load_specs(path: str = SPECS_FILE) -> Dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_design(template: str, params: List[Dict]) -> List[str]:
    """Check that all {placeholders} in design.cir match parameters.csv."""
    errors = []
    circuit_lines = []
    in_control = False
    for line in template.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith(".control"):
            in_control = True
        if not in_control and not stripped.startswith("*"):
            circuit_lines.append(line)
        if stripped.lower().startswith(".endc"):
            in_control = False
    circuit_text = "\n".join(circuit_lines)
    placeholders = set(re.findall(r'\{(\w+)\}', circuit_text))
    param_names = {p["name"] for p in params}

    for m in sorted(placeholders - param_names):
        errors.append(f"Placeholder {{{m}}} in design.cir has no entry in parameters.csv")
    for u in sorted(param_names - placeholders):
        errors.append(f"Parameter '{u}' in parameters.csv is not used in design.cir")

    return errors


# ---------------------------------------------------------------------------
# NGSpice simulation
# ---------------------------------------------------------------------------

def format_netlist(template: str, param_values: Dict[str, float]) -> str:
    """Format netlist with parameter values using regex substitution."""
    def _replace(match):
        key = match.group(1)
        if key in param_values:
            return str(param_values[key])
        return match.group(0)
    return re.sub(r'\{(\w+)\}', _replace, template)


def run_simulation(template: str, param_values: Dict[str, float],
                   idx: int, tmp_dir: str) -> Dict:
    """Format netlist and run ngspice."""
    try:
        netlist = format_netlist(template, param_values)
    except Exception as e:
        return {"idx": idx, "error": f"format error: {e}", "measurements": {}}

    path = os.path.join(tmp_dir, f"sim_{idx}.cir")
    with open(path, "w") as f:
        f.write(netlist)

    try:
        result = subprocess.run(
            [NGSPICE, "-b", path],
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"idx": idx, "error": "timeout", "measurements": {}}
    except Exception as e:
        return {"idx": idx, "error": str(e), "measurements": {}}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if "RESULT_DONE" not in output:
        return {"idx": idx, "error": "no_RESULT_DONE", "measurements": {},
                "output_tail": output[-500:]}

    measurements = parse_ngspice_output(output)

    # Compute CMRR from individual gains if not already present
    if "RESULT_CMRR_DB" not in measurements:
        dc_g = measurements.get("dc_gain_db")
        cm_g = measurements.get("cm_gain_db")
        if dc_g is not None and cm_g is not None and dc_g != cm_g:
            measurements["RESULT_CMRR_DB"] = dc_g - cm_g

    return {"idx": idx, "error": None, "measurements": measurements}


def parse_ngspice_output(output: str) -> Dict[str, float]:
    """Extract all measurements from ngspice output."""
    m = {}
    for line in output.split("\n"):
        # Parse RESULT_* echo lines
        if "RESULT_" in line and "RESULT_DONE" not in line:
            match = re.search(r'(RESULT_\w+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
            if match:
                m[match.group(1)] = float(match.group(2))

        # Parse meas output: "name = value"
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith((".", "*", "+")):
            parts = stripped.split("=", 1)
            if len(parts) == 2:
                name = parts[0].strip()
                val_match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', parts[1])
                if val_match and name and len(name) < 40 and not name.startswith("("):
                    try:
                        m[name] = float(val_match.group(1))
                    except ValueError:
                        pass
    return m


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

def compute_cost(measurements: Dict[str, float], specs: Dict) -> float:
    """Compute cost from measurements. Lower is better."""
    if not measurements:
        return 1e6

    cost = 0.0
    spec_defs = specs["measurements"]

    # Map measurement names to spec names
    measurement_map = {
        "dc_gain_db": ["dc_gain_db", "DC_GAIN_DB", "RESULT_DC_GAIN_DB"],
        "gbw_hz": ["gbw_hz", "GBW_HZ", "RESULT_GBW_HZ"],
        "phase_margin_deg": ["RESULT_PHASE_MARGIN", "phase_margin_deg", "phase_margin"],
        "power_mw": ["RESULT_POWER_MW", "power_mw", "POWER_MW"],
        "output_swing_v": ["RESULT_SWING", "output_swing_v", "swing"],
        "cmrr_db": ["RESULT_CMRR_DB", "cmrr_db", "CMRR_DB"],
    }

    for spec_name, spec_def in spec_defs.items():
        target_str = spec_def["target"]
        weight = spec_def["weight"] / 100.0

        if target_str.startswith(">"):
            direction = "above"
            target_val = float(target_str[1:])
        elif target_str.startswith("<"):
            direction = "below"
            target_val = float(target_str[1:])
        else:
            continue

        # Find measured value
        measured = None
        for key in measurement_map.get(spec_name, [spec_name]):
            if key in measurements:
                measured = measurements[key]
                break

        if measured is None:
            cost += weight * 1000
            continue

        if direction == "above":
            if measured >= target_val:
                ratio = measured / max(abs(target_val), 1e-12)
                cost -= weight * min(ratio - 1.0, 1.0) * 10
            else:
                gap = (target_val - measured) / max(abs(target_val), 1e-12)
                cost += weight * gap ** 2 * 500
        else:
            if measured <= target_val:
                ratio = measured / max(abs(target_val), 1e-12)
                cost -= weight * min(1.0 - ratio, 1.0) * 10
            else:
                gap = (measured - target_val) / max(abs(target_val), 1e-12)
                cost += weight * gap ** 2 * 500

    # Bias point penalties (NMOS saturation check)
    vto_n = 0.7
    vto_p = -0.7
    for suffix in ["1", "5", "7"]:
        vgs = measurements.get(f"RESULT_VGS{suffix}")
        vds = measurements.get(f"RESULT_VDS{suffix}")
        if vgs is not None and vds is not None:
            if vgs < vto_n:
                cost += 100
            elif vds < vgs - vto_n:
                cost += 50

    for suffix in ["6"]:
        vgs = measurements.get(f"RESULT_VGS{suffix}")
        vds = measurements.get(f"RESULT_VDS{suffix}")
        if vgs is not None and vds is not None:
            if vgs > vto_p:
                cost += 100
            elif vds > vgs - vto_p:
                cost += 50

    return cost


# ---------------------------------------------------------------------------
# Parallel evaluator
# ---------------------------------------------------------------------------

def eval_batch_local(template: str, param_dicts: List[Dict[str, float]],
                     specs: Dict, n_workers: int) -> Dict:
    """Evaluate a batch of parameter sets locally."""
    tmp_dir = tempfile.mkdtemp(prefix="opamp_de_")
    n = len(param_dicts)
    results = [None] * n

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(run_simulation, template, p, i, tmp_dir): i
            for i, p in enumerate(param_dicts)
        }
        for future in as_completed(futures):
            r = future.result()
            results[r["idx"]] = r

    metrics = []
    for r in results:
        if r is None or r.get("error"):
            metrics.append(1e6)
        else:
            metrics.append(compute_cost(r["measurements"], specs))

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return {"metrics": metrics}


def eval_batch_remote(template: str, param_dicts: List[Dict[str, float]],
                      specs: Dict, server_url: str) -> Dict:
    """Evaluate via remote sim server."""
    import requests
    metric_func_code = _build_metric_func_code(specs)
    payload = {
        "parameters": param_dicts,
        "circuit_template": template,
        "metric_func": metric_func_code,
    }
    r = requests.post(f"{server_url}/evaluate", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def _build_metric_func_code(specs: Dict) -> str:
    """Generate compute_metric function code for remote server."""
    specs_json = json.dumps(specs)
    return f'''
import json
_specs = json.loads({repr(specs_json)})
def compute_metric(measurements):
    if not measurements:
        return 1e6
    cost = 0.0
    for spec_name, spec_def in _specs["measurements"].items():
        target_str = spec_def["target"]
        weight = spec_def["weight"] / 100.0
        if target_str.startswith(">"):
            direction, target_val = "above", float(target_str[1:])
        elif target_str.startswith("<"):
            direction, target_val = "below", float(target_str[1:])
        else:
            continue
        measured = None
        for key in [spec_name, spec_name.upper(), f"RESULT_{{spec_name.upper()}}"]:
            if key in measurements:
                measured = measurements[key]
                break
        if measured is None:
            cost += weight * 1000
            continue
        if direction == "above":
            if measured >= target_val:
                cost -= weight * min(measured / max(abs(target_val), 1e-12) - 1.0, 1.0) * 10
            else:
                gap = (target_val - measured) / max(abs(target_val), 1e-12)
                cost += weight * gap ** 2 * 500
        else:
            if measured <= target_val:
                cost -= weight * min(1.0 - measured / max(abs(target_val), 1e-12), 1.0) * 10
            else:
                gap = (measured - target_val) / max(abs(target_val), 1e-12)
                cost += weight * gap ** 2 * 500
    return cost
'''


# ---------------------------------------------------------------------------
# DE runner
# ---------------------------------------------------------------------------

def run_de(template: str, params: List[Dict], specs: Dict,
           n_workers: int = 0, server_url: str = "",
           quick: bool = False) -> Dict:
    """Run Differential Evolution to optimize parameters."""

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from de.engine import DifferentialEvolution, load_parameters as de_load_parameters

    # Convert param list to DE param dict via temp CSV
    tmp_csv = os.path.join(tempfile.gettempdir(), "_de_params.csv")
    with open(tmp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "min", "max", "scale"])
        for p in params:
            w.writerow([p["name"], p["min"], p["max"], p.get("scale", "lin")])
    de_params = de_load_parameters(tmp_csv)
    os.unlink(tmp_csv)

    n_params = len(params)
    pop_size = max(100, 5 * n_params) if not quick else max(30, 2 * n_params)
    patience = 50 if not quick else 10
    min_iter = 30 if not quick else 5
    max_iter = 5000 if not quick else 50

    if not n_workers:
        n_workers = os.cpu_count() or 8

    if server_url:
        def eval_func(parameters, **kwargs):
            return eval_batch_remote(template, parameters, specs, server_url)
    else:
        def eval_func(parameters, **kwargs):
            return eval_batch_local(template, parameters, specs, n_workers)

    print(f"DE: {n_params} params, pop={pop_size}, patience={patience}, "
          f"workers={n_workers if not server_url else 'remote'}, adaptive budget")

    de = DifferentialEvolution(
        params=de_params,
        eval_func=eval_func,
        pop_size=pop_size,
        opt_dir="min",
        min_iterations=min_iter,
        max_iterations=max_iter,
        metric_threshold=-50.0,
        patience=patience,
        F1=0.7, F2=0.3, F3=0.1, CR=0.9,
    )

    return de.run()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_measurements(measurements: Dict[str, float], specs: Dict) -> Tuple[float, Dict]:
    """Score measurements against specs. Returns (score 0-1, details)."""
    details = {}
    total_weight = 0
    weighted_score = 0

    measurement_map = {
        "dc_gain_db": ["dc_gain_db", "DC_GAIN_DB", "RESULT_DC_GAIN_DB"],
        "gbw_hz": ["gbw_hz", "GBW_HZ", "RESULT_GBW_HZ"],
        "phase_margin_deg": ["RESULT_PHASE_MARGIN", "phase_margin_deg"],
        "power_mw": ["RESULT_POWER_MW", "power_mw"],
        "output_swing_v": ["RESULT_SWING", "output_swing_v"],
        "cmrr_db": ["RESULT_CMRR_DB", "cmrr_db"],
    }

    for spec_name, spec_def in specs["measurements"].items():
        target_str = spec_def["target"]
        weight = spec_def["weight"]
        unit = spec_def.get("unit", "")
        total_weight += weight

        if target_str.startswith(">"):
            target_val = float(target_str[1:])
            direction = "above"
        elif target_str.startswith("<"):
            target_val = float(target_str[1:])
            direction = "below"
        else:
            continue

        measured = None
        for key in measurement_map.get(spec_name, [spec_name]):
            if key in measurements:
                measured = measurements[key]
                break

        if measured is None:
            details[spec_name] = {
                "measured": None, "target": target_str, "met": False,
                "score": 0, "unit": unit
            }
            continue

        if direction == "above":
            if measured >= target_val:
                spec_score = 1.0
            elif target_val == 0:
                spec_score = 0.0
            else:
                spec_score = max(0, measured / target_val)
        else:
            if measured <= target_val:
                spec_score = 1.0
            elif measured == 0:
                spec_score = 0.0
            else:
                spec_score = max(0, target_val / measured)

        met = spec_score >= 1.0
        weighted_score += weight * spec_score
        details[spec_name] = {
            "measured": measured, "target": target_str, "met": met,
            "score": spec_score, "unit": unit
        }

    overall = weighted_score / total_weight if total_weight > 0 else 0
    return overall, details


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_plots(template: str, best_params: Dict[str, float],
                   specs: Dict, plots_dir: str):
    """Generate Bode plot and output swing plot for GitHub."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot generation")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # Format netlist with best parameters
    netlist = format_netlist(template, best_params)

    # Extract circuit (remove .control block and .end)
    circuit = re.sub(r'\.control.*?\.endc', '', netlist, flags=re.DOTALL | re.IGNORECASE)
    circuit = re.sub(r'\.end\s*$', '', circuit, flags=re.MULTILINE).strip()

    tmp_dir = tempfile.mkdtemp(prefix="opamp_plot_")
    ac_file = os.path.join(tmp_dir, "ac_data")
    dc_file = os.path.join(tmp_dir, "dc_data")

    # Build plot netlist with wrdata export
    plot_control = f"""
.control
op

* AC analysis for Bode plot
ac dec 200 1 1G
wrdata {ac_file} vdb(out) vp(out)

* DC sweep for output swing
dc Vinp -14 14 0.01
wrdata {dc_file} v(out)

quit
.endc

.end
"""
    plot_netlist = circuit + "\n" + plot_control
    plot_cir = os.path.join(tmp_dir, "plot.cir")
    with open(plot_cir, "w") as f:
        f.write(plot_netlist)

    # Run ngspice
    try:
        result = subprocess.run(
            [NGSPICE, "-b", plot_cir],
            capture_output=True, text=True, timeout=120
        )
    except Exception as e:
        print(f"Plot simulation failed: {e}")
        return

    # Parse AC data and generate Bode plot
    try:
        ac_data = _parse_wrdata(ac_file)
        if ac_data is not None and len(ac_data) > 0:
            freq = ac_data[:, 0]
            gain_db = ac_data[:, 1]
            phase_deg = ac_data[:, 2] if ac_data.shape[1] > 2 else None

            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.semilogx(freq, gain_db, 'b-', linewidth=1.5, label='Gain')
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Gain (dB)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True, which='both', alpha=0.3)
            ax1.axhline(y=0, color='b', linestyle='--', alpha=0.5)

            # Mark GBW
            zero_crossings = np.where(np.diff(np.sign(gain_db)))[0]
            if len(zero_crossings) > 0:
                gbw_idx = zero_crossings[0]
                gbw_freq = freq[gbw_idx]
                ax1.axvline(x=gbw_freq, color='r', linestyle=':', alpha=0.7,
                            label=f'GBW = {gbw_freq:.2e} Hz')

            if phase_deg is not None:
                ax2 = ax1.twinx()
                ax2.semilogx(freq, phase_deg, 'r-', linewidth=1.0, alpha=0.7, label='Phase')
                ax2.set_ylabel('Phase (deg)', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                ax2.axhline(y=-180, color='r', linestyle='--', alpha=0.3)

            ax1.set_title('Open-Loop Bode Plot')
            ax1.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "bode.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {plots_dir}/bode.png")
    except Exception as e:
        print(f"  Bode plot failed: {e}")

    # Parse DC data and generate output swing plot
    try:
        dc_data = _parse_wrdata(dc_file)
        if dc_data is not None and len(dc_data) > 0:
            vin = dc_data[:, 0]
            vout = dc_data[:, 1]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(vin, vout, 'b-', linewidth=1.5)
            ax.set_xlabel('Differential Input (V)')
            ax.set_ylabel('Output (V)')
            ax.set_title('DC Transfer Curve (Output Swing)')
            ax.grid(True, alpha=0.3)

            vout_min = np.min(vout)
            vout_max = np.max(vout)
            swing = vout_max - vout_min
            ax.axhline(y=vout_max, color='r', linestyle='--', alpha=0.5,
                        label=f'Max = {vout_max:.2f}V')
            ax.axhline(y=vout_min, color='r', linestyle='--', alpha=0.5,
                        label=f'Min = {vout_min:.2f}V')
            ax.legend()
            ax.set_title(f'DC Transfer Curve — Swing = {swing:.1f}V')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "output_swing.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {plots_dir}/output_swing.png")
    except Exception as e:
        print(f"  Output swing plot failed: {e}")

    # Cleanup
    for f_path in [ac_file, dc_file, plot_cir]:
        try:
            os.unlink(f_path)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass


def _parse_wrdata(filepath: str) -> Optional[np.ndarray]:
    """Parse ngspice wrdata output file. Returns Nx3 array or None."""
    if not os.path.exists(filepath):
        return None
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('*'):
                continue
            parts = line.split()
            try:
                vals = [float(x) for x in parts]
                if len(vals) >= 2:
                    rows.append(vals)
            except ValueError:
                continue
    if not rows:
        return None
    return np.array(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(best_params: Dict, measurements: Dict, score: float,
                 details: Dict, specs: Dict, de_result: Dict, elapsed: float):
    """Print human-readable evaluation report."""
    print(f"\n{'='*70}")
    print(f"  EVALUATION REPORT — {specs.get('name', 'Circuit')}")
    print(f"{'='*70}")
    print(f"\n  Score: {score:.2f} / 1.00  |  Time: {elapsed:.1f}s")
    print(f"  DE converged: {de_result.get('converged', 'N/A')}  |  "
          f"Iterations: {de_result.get('iterations', 'N/A')}  |  "
          f"Diversity: {de_result.get('diversity', 0):.4f}")
    print(f"  Stop reason: {de_result.get('stop_reason', 'N/A')}")

    specs_met = sum(1 for d in details.values() if d.get("met"))
    specs_total = len(details)
    print(f"\n  Specs met: {specs_met}/{specs_total}")

    print(f"\n  {'Spec':<22} {'Target':>10} {'Measured':>12} {'Unit':>6} {'Status':>8} {'Score':>6}")
    print(f"  {'-'*68}")

    for spec_name, d in details.items():
        target = d["target"]
        measured = d["measured"]
        unit = d["unit"]
        met = d["met"]
        s = d["score"]

        if measured is None:
            m_str = "N/A"
        elif abs(measured) > 1e6:
            m_str = f"{measured:.2e}"
        elif abs(measured) < 0.01:
            m_str = f"{measured:.2e}"
        else:
            m_str = f"{measured:.2f}"

        status = "PASS" if met else "FAIL"
        print(f"  {spec_name:<22} {target:>10} {m_str:>12} {unit:>6} {status:>8} {s:>5.2f}")

    print(f"\n  Best Parameters:")
    for name, val in sorted(best_params.items()):
        print(f"    {name:<12} = {val:.4e}")
    print(f"\n{'='*70}\n")

    return specs_met, specs_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate op-amp design")
    parser.add_argument("--server", type=str, default="",
                        help="Remote sim server URL")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of local ngspice workers")
    parser.add_argument("--quick", action="store_true",
                        help="Quick evaluation (small pop, low patience)")
    args = parser.parse_args()

    # Load files
    print("Loading design...")
    template = load_design()
    params = load_parameters()
    specs = load_specs()

    # Validate
    errors = validate_design(template, params)
    if errors:
        print("\nVALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)

    print(f"Design: {specs.get('name', 'Unknown')}")
    print(f"Parameters: {len(params)}")
    print(f"Specs: {len(specs['measurements'])}")
    print()

    # Run DE
    t0 = time.time()
    de_result = run_de(
        template=template,
        params=params,
        specs=specs,
        n_workers=args.workers,
        server_url=args.server,
        quick=args.quick,
    )
    elapsed = time.time() - t0

    best_params = de_result["best_parameters"]

    # Run one final sim with best params
    tmp_dir = tempfile.mkdtemp(prefix="opamp_final_")
    final = run_simulation(template, best_params, 0, tmp_dir)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    if final.get("error"):
        print(f"\nFinal simulation failed: {final['error']}")
        measurements = {}
    else:
        measurements = final["measurements"]

    # Score
    score, details = score_measurements(measurements, specs)

    # Report
    specs_met, specs_total = print_report(
        best_params, measurements, score, details, specs, de_result, elapsed)

    # Generate plots
    print("Generating plots...")
    generate_plots(template, best_params, specs, PLOTS_DIR)

    # Save results
    with open("best_parameters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "value"])
        for name, val in sorted(best_params.items()):
            w.writerow([name, val])

    with open("measurements.json", "w") as f:
        json.dump({
            "measurements": measurements,
            "score": score,
            "details": details,
            "parameters": best_params,
            "de_result": {
                "converged": de_result.get("converged"),
                "iterations": de_result.get("iterations"),
                "diversity": de_result.get("diversity"),
                "stop_reason": de_result.get("stop_reason"),
                "best_metric": de_result.get("best_metric"),
            },
        }, f, indent=2)

    print(f"\nSaved: best_parameters.csv, measurements.json, {PLOTS_DIR}/")
    print(f"Score: {score:.2f} | Specs met: {specs_met}/{specs_total} | "
          f"Converged: {de_result.get('converged')}")

    return score


if __name__ == "__main__":
    score = main()
    sys.exit(0 if score >= 0.9 else 1)
