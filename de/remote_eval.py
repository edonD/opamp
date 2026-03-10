"""Remote evaluator — sends parameter batches to sim server via HTTP.

Replaces ngspice_eval.py's local ProcessPool with a remote call.
The DE engine doesn't know the difference.
"""

import requests
import time
from typing import Dict, List, Optional


class RemoteEvaluator:
    """Evaluator that sends work to a remote sim server.

    Usage:
        evaluator = RemoteEvaluator("http://sim-node:8000")
        evaluator.configure(circuit_template=my_template, metric_func=my_func)

        # Use as DE eval_func:
        de = DifferentialEvolution(
            eval_func=evaluator.evaluate,
            ...
        )
    """

    def __init__(self, server_url: str, timeout: int = 600):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._circuit_template: Optional[str] = None
        self._metric_func: Optional[str] = None

        # Verify connection
        self._check_health()

    def _check_health(self):
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5)
            r.raise_for_status()
            info = r.json()
            print(f"[RemoteEval] Connected to {self.server_url}")
            print(f"[RemoteEval] Workers: {info['n_workers']}, NGSpice: {info['ngspice']}")
        except Exception as e:
            print(f"[RemoteEval] WARNING: Cannot reach {self.server_url}: {e}")

    def configure(self, circuit_template: str = "", metric_func: str = ""):
        """Send circuit template and metric function to the server.

        Args:
            circuit_template: SPICE netlist with {param} placeholders
            metric_func: Python code defining compute_metric(measurements) -> float
        """
        self._circuit_template = circuit_template or self._circuit_template
        self._metric_func = metric_func or self._metric_func

        # Also push template to server via /configure
        if circuit_template:
            r = requests.post(f"{self.server_url}/configure",
                              json={"circuit_template": circuit_template},
                              timeout=10)
            r.raise_for_status()
            print(f"[RemoteEval] Template configured ({len(circuit_template)} chars)")

    def evaluate(self, parameters: List[Dict], **kwargs) -> Dict:
        """Send parameter batch to sim server, get metrics back.

        This is the eval_func signature that DE expects.
        """
        t0 = time.time()

        payload = {
            "parameters": parameters,
        }

        # Include template and metric func if set (sent per-request for flexibility)
        if self._circuit_template:
            payload["circuit_template"] = self._circuit_template
        if self._metric_func:
            payload["metric_func"] = self._metric_func

        try:
            r = requests.post(
                f"{self.server_url}/evaluate",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.Timeout:
            print(f"[RemoteEval] TIMEOUT after {self.timeout}s")
            return {"metrics": [1e6] * len(parameters)}
        except Exception as e:
            print(f"[RemoteEval] ERROR: {e}")
            return {"metrics": [1e6] * len(parameters)}

        network_time = time.time() - t0
        server_time = data.get("total_time", 0)
        n_failed = data.get("n_failed", 0)

        if n_failed > 0:
            print(f"[RemoteEval] {len(parameters)} sims | "
                  f"server: {server_time:.2f}s | network: {network_time:.2f}s | "
                  f"failed: {n_failed}")

        return {"metrics": data["metrics"], "measurements": data.get("measurements", [])}
