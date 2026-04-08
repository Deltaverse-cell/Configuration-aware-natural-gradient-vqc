
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def real_scalar(value: complex | float | np.ndarray) -> float:
    scalar = np.asarray(value).item()
    return float(np.real(scalar))


def ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0.0], [0.0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def kron_all(mats: Sequence[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def single_qubit_op(nq: int, q: int, gate: np.ndarray) -> np.ndarray:
    mats = [I2 for _ in range(nq)]
    mats[q] = gate
    return kron_all(mats)


def pauli_op(nq: int, s: str) -> np.ndarray:
    lookup = {"I": I2, "X": X, "Y": Y, "Z": Z}
    return kron_all([lookup[ch] for ch in s])


def rzz(theta: float) -> np.ndarray:
    a = np.exp(-1j * theta / 2)
    b = np.exp(1j * theta / 2)
    return np.diag([a, b, b, a]).astype(np.complex128)


def two_qubit_op(nq: int, q1: int, q2: int, gate4: np.ndarray) -> np.ndarray:
    if q1 == q2:
        raise ValueError("q1 and q2 must be distinct")
    if q1 > q2:
        q1, q2 = q2, q1

    dim = 2**nq
    out = np.zeros((dim, dim), dtype=np.complex128)
    for col in range(dim):
        bits = [(col >> (nq - 1 - k)) & 1 for k in range(nq)]
        src_idx = bits[q1] * 2 + bits[q2]
        for out_pair in range(4):
            amp = gate4[out_pair, src_idx]
            if abs(amp) < 1e-15:
                continue
            nb = bits.copy()
            nb[q1] = (out_pair >> 1) & 1
            nb[q2] = out_pair & 1
            row = 0
            for b in nb:
                row = (row << 1) | b
            out[row, col] += amp
    return out


@dataclasses.dataclass(frozen=True)
class PauliTerm:
    coeff: float
    pauli: str


def tfim_hamiltonian_3q(h: float = 0.7) -> List[PauliTerm]:
    return [
        PauliTerm(1.0, "ZZI"),
        PauliTerm(1.0, "IZZ"),
        PauliTerm(h, "XII"),
        PauliTerm(h, "IXI"),
        PauliTerm(h, "IIX"),
    ]


def heisenberg_hamiltonian_3q() -> List[PauliTerm]:
    return [
        PauliTerm(1.0, "XXI"),
        PauliTerm(1.0, "YYI"),
        PauliTerm(1.0, "ZZI"),
        PauliTerm(1.0, "IXX"),
        PauliTerm(1.0, "IYY"),
        PauliTerm(1.0, "IZZ"),
    ]


@dataclasses.dataclass
class Configuration:
    name: str
    edge_order_per_layer: List[List[Tuple[int, int]]]
    rotation_mode: str
    param_bias: np.ndarray
    entangler_rel_error: float


@dataclasses.dataclass
class TaskSpec:
    name: str
    h_terms: List[PauliTerm]


@dataclasses.dataclass
class Counters:
    state_evals: int = 0


class FamilyExperiment:
    def __init__(self, n_qubits: int = 3, depth: int = 2, base_rzz: float = math.pi / 2):
        self.n_qubits = n_qubits
        self.depth = depth
        self.base_rzz = base_rzz
        self.n_params = 2 * n_qubits * depth
        self._pauli_cache: Dict[str, np.ndarray] = {}

    def build_family(self, variability_level: float, seed: int) -> List[Configuration]:
        rng = np.random.default_rng(seed)
        biases = [rng.normal(0.0, variability_level, size=self.n_params) for _ in range(3)]
        return [
            Configuration(
                name="cfg_forward",
                edge_order_per_layer=[[(0, 1), (1, 2)] for _ in range(self.depth)],
                rotation_mode="ry_rz",
                param_bias=biases[0],
                entangler_rel_error=+0.5 * variability_level,
            ),
            Configuration(
                name="cfg_reverse",
                edge_order_per_layer=[[(1, 2), (0, 1)] for _ in range(self.depth)],
                rotation_mode="ry_rz",
                param_bias=biases[1],
                entangler_rel_error=-0.5 * variability_level,
            ),
            Configuration(
                name="cfg_rotflip",
                edge_order_per_layer=[[(0, 1), (1, 2)] for _ in range(self.depth)],
                rotation_mode="rz_ry",
                param_bias=biases[2],
                entangler_rel_error=0.0,
            ),
        ]

    def _statevector(self, theta: np.ndarray, cfg: Configuration, counters: Optional[Counters] = None) -> np.ndarray:
        if counters is not None:
            counters.state_evals += 1

        psi = np.zeros(2**self.n_qubits, dtype=np.complex128)
        psi[0] = 1.0
        te = theta + cfg.param_bias
        idx = 0
        for layer in range(self.depth):
            for q in range(self.n_qubits):
                a, b = te[idx], te[idx + 1]
                idx += 2
                Uq = rz(b) @ ry(a) if cfg.rotation_mode == "ry_rz" else ry(b) @ rz(a)
                psi = single_qubit_op(self.n_qubits, q, Uq) @ psi

            angle = self.base_rzz * (1.0 + cfg.entangler_rel_error)
            gate = rzz(angle)
            for u, v in cfg.edge_order_per_layer[layer]:
                psi = two_qubit_op(self.n_qubits, u, v, gate) @ psi

        return psi

    def _pauli_matrix(self, s: str) -> np.ndarray:
        if s not in self._pauli_cache:
            self._pauli_cache[s] = pauli_op(self.n_qubits, s)
        return self._pauli_cache[s]

    def term_expectation(self, psi: np.ndarray, pauli_string: str) -> float:
        P = self._pauli_matrix(pauli_string)
        return real_scalar(np.vdot(psi, P @ psi))

    def energy(self, theta: np.ndarray, cfg: Configuration, task: TaskSpec, counters: Optional[Counters] = None) -> float:
        psi = self._statevector(theta, cfg, counters)
        return float(sum(term.coeff * self.term_expectation(psi, term.pauli) for term in task.h_terms))

    def family_losses(
        self,
        theta: np.ndarray,
        family: List[Configuration],
        task: TaskSpec,
        counters: Optional[Counters] = None,
    ) -> np.ndarray:
        return np.array([self.energy(theta, cfg, task, counters) for cfg in family], dtype=float)

    def gradient_param_shift(
        self,
        theta: np.ndarray,
        cfg: Configuration,
        task: TaskSpec,
        counters: Optional[Counters] = None,
    ) -> np.ndarray:
        grad = np.zeros_like(theta)
        shift = np.pi / 2
        for p in range(theta.size):
            tp = theta.copy()
            tm = theta.copy()
            tp[p] += shift
            tm[p] -= shift
            grad[p] = 0.5 * (
                self.energy(tp, cfg, task, counters) - self.energy(tm, cfg, task, counters)
            )
        return grad

    def qgt_real(
        self,
        theta: np.ndarray,
        cfg: Configuration,
        counters: Optional[Counters] = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        psi = self._statevector(theta, cfg, counters)
        derivs = []
        for p in range(theta.size):
            tp = theta.copy()
            tm = theta.copy()
            tp[p] += eps
            tm[p] -= eps
            psi_p = self._statevector(tp, cfg, counters)
            psi_m = self._statevector(tm, cfg, counters)
            derivs.append((psi_p - psi_m) / (2 * eps))

        d = theta.size
        G = np.zeros((d, d), dtype=float)
        for a in range(d):
            for b in range(a, d):
                term = np.vdot(derivs[a], derivs[b]) - np.vdot(derivs[a], psi) * np.vdot(psi, derivs[b])
                val = float(np.real(term))
                G[a, b] = val
                G[b, a] = val
        return 0.5 * (G + G.T)


def safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def qng_step(G: np.ndarray, g: np.ndarray, lr: float, lam: float) -> np.ndarray:
    return -lr * safe_solve(G + lam * np.eye(G.shape[0]), g)


def block_qng_step(G: np.ndarray, g: np.ndarray, lr: float, lam: float, block_size: int = 2) -> np.ndarray:
    d = G.shape[0]
    out = np.zeros(d)
    for s in range(0, d, block_size):
        e = min(d, s + block_size)
        out[s:e] = -lr * safe_solve(G[s:e, s:e] + lam * np.eye(e - s), g[s:e])
    return out


def metric_condition(G: np.ndarray, lam: float) -> float:
    vals = np.linalg.eigvalsh(G + lam * np.eye(G.shape[0]))
    vals = np.maximum(vals, 1e-12)
    return float(vals.max() / vals.min())


@dataclasses.dataclass
class RunConfig:
    lr_gd: float = 0.08
    lr_qng: float = 0.08
    lam: float = 1e-3
    max_iters: int = 60
    grad_tol: float = 1e-7
    clip_step_norm: float = 1.0


class FAVORunner:
    def __init__(self, exp: FamilyExperiment, cfg: RunConfig):
        self.exp = exp
        self.cfg = cfg

    def _clip_step(self, step: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(step))
        if norm > self.cfg.clip_step_norm:
            step = step * (self.cfg.clip_step_norm / norm)
        return step

    def run_task_seed(self, task: TaskSpec, family: List[Configuration], seed: int):
        rng = np.random.default_rng(seed)
        theta0 = rng.uniform(-0.3, 0.3, size=self.exp.n_params)

        methods = ["GD", "QNG", "BlockQNG", "UF-GD", "UF-QNG-Mix"]
        run_rows = []
        hist_rows = []

        for method in methods:
            theta = theta0.copy()
            counters = Counters()
            converged_early = False
            last_iter = -1

            for it in range(self.cfg.max_iters):
                last_iter = it
                losses = self.exp.family_losses(theta, family, task, counters)
                family_loss = float(np.mean(losses))
                family_var = float(np.var(losses))
                worst_loss = float(np.max(losses))

                grads = [self.exp.gradient_param_shift(theta, cfg, task, counters) for cfg in family]
                qgts = [self.exp.qgt_real(theta, cfg, counters) for cfg in family]

                grad_ref = grads[0]
                qgt_ref = qgts[0]
                grad_fam = np.mean(np.stack(grads, axis=0), axis=0)
                qgt_fam = np.mean(np.stack(qgts, axis=0), axis=0)

                grad_ref_norm = float(np.linalg.norm(grad_ref))
                grad_fam_norm = float(np.linalg.norm(grad_fam))

                if method == "GD":
                    if grad_ref_norm < self.cfg.grad_tol:
                        converged_early = True
                        break
                    step = -self.cfg.lr_gd * grad_ref
                    selected_config = "cfg_forward"
                    effective_condition = float("nan")
                elif method == "QNG":
                    if grad_ref_norm < self.cfg.grad_tol:
                        converged_early = True
                        break
                    step = qng_step(qgt_ref, grad_ref, self.cfg.lr_qng, self.cfg.lam)
                    selected_config = "cfg_forward"
                    effective_condition = metric_condition(qgt_ref, self.cfg.lam)
                elif method == "BlockQNG":
                    if grad_ref_norm < self.cfg.grad_tol:
                        converged_early = True
                        break
                    step = block_qng_step(qgt_ref, grad_ref, self.cfg.lr_qng, self.cfg.lam)
                    selected_config = "cfg_forward"
                    effective_condition = metric_condition(qgt_ref, self.cfg.lam)
                elif method == "UF-GD":
                    if grad_fam_norm < self.cfg.grad_tol:
                        converged_early = True
                        break
                    step = -self.cfg.lr_gd * grad_fam
                    selected_config = "family_average"
                    effective_condition = float("nan")
                elif method == "UF-QNG-Mix":
                    if grad_fam_norm < self.cfg.grad_tol:
                        converged_early = True
                        break
                    step = qng_step(qgt_fam, grad_fam, self.cfg.lr_qng, self.cfg.lam)
                    selected_config = "family_average"
                    effective_condition = metric_condition(qgt_fam, self.cfg.lam)
                else:
                    raise ValueError(f"Unknown method: {method}")

                step = self._clip_step(step)
                theta = theta + step

                hist_rows.append(
                    {
                        "task": task.name,
                        "seed": seed,
                        "method": method,
                        "iteration": it,
                        "family_loss_true": family_loss,
                        "family_var_true": family_var,
                        "worst_config_loss_true": worst_loss,
                        "selected_config": selected_config,
                        "step_norm": float(np.linalg.norm(step)),
                        "grad_ref_norm": grad_ref_norm,
                        "grad_family_norm": grad_fam_norm,
                        "effective_condition": effective_condition,
                        **{f"loss_true__{cfg.name}": float(val) for cfg, val in zip(family, losses)},
                    }
                )

            final_losses = self.exp.family_losses(theta, family, task, counters)
            run_rows.append(
                {
                    "task": task.name,
                    "seed": seed,
                    "method": method,
                    "final_family_loss_true": float(np.mean(final_losses)),
                    "final_family_var_true": float(np.var(final_losses)),
                    "final_worst_config_loss_true": float(np.max(final_losses)),
                    "iterations_completed": max(last_iter + 1, 0),
                    "converged_early": converged_early,
                    "state_evals": counters.state_evals,
                    **{f"final_loss_true__{cfg.name}": float(val) for cfg, val in zip(family, final_losses)},
                }
            )

        return pd.DataFrame(run_rows), pd.DataFrame(hist_rows)


def summarize_results(run_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_family_loss_true",
        "final_family_var_true",
        "final_worst_config_loss_true",
        "iterations_completed",
        "state_evals",
    ]
    out = run_df.groupby(["task", "method"])[metrics].agg(["mean", "std", "median"]).reset_index()
    out.columns = ["__".join([c for c in col if c]) for col in out.columns.to_flat_index()]
    return out


def paired_differences(run_df: pd.DataFrame, reference_method: str) -> pd.DataFrame:
    rows = []
    for task in sorted(run_df["task"].unique()):
        task_df = run_df[run_df["task"] == task]
        ref = task_df[task_df["method"] == reference_method].set_index("seed")
        for method in sorted(task_df["method"].unique()):
            if method == reference_method:
                continue
            other = task_df[task_df["method"] == method].set_index("seed")
            for seed in sorted(set(ref.index) & set(other.index)):
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "reference_method": reference_method,
                        "method": method,
                        "delta_family_loss_true": float(other.loc[seed, "final_family_loss_true"] - ref.loc[seed, "final_family_loss_true"]),
                        "delta_family_var_true": float(other.loc[seed, "final_family_var_true"] - ref.loc[seed, "final_family_var_true"]),
                        "delta_worst_config_loss_true": float(other.loc[seed, "final_worst_config_loss_true"] - ref.loc[seed, "final_worst_config_loss_true"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_task_history(hist_df: pd.DataFrame, task_name: str, output_dir: Path) -> None:
    t = hist_df[hist_df["task"] == task_name].copy()
    if t.empty:
        return

    specs = [
        ("family_loss_true", "True family loss", "family_loss_vs_iteration.png"),
        ("family_var_true", "True family variance", "family_variance_vs_iteration.png"),
        ("worst_config_loss_true", "Worst configuration loss", "worst_config_loss_vs_iteration.png"),
    ]
    for metric, ylabel, fname in specs:
        plt.figure(figsize=(8, 5))
        for method, sub in t.groupby("method"):
            med = sub.groupby("iteration")[metric].median()
            plt.plot(med.index, med.values, label=method)
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.title(f"{task_name}: {ylabel}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"{task_name}__{fname}", dpi=160)
        plt.close()


def plot_variability_sweep(run_df: pd.DataFrame, output_dir: Path) -> None:
    run_df = run_df.copy()
    run_df["task_base"] = run_df["task"].str.split("__var_").str[0]
    specs = [
        ("final_family_loss_true", "Final true family loss", "variability_sweep_family_loss.png"),
        ("final_family_var_true", "Final true family variance", "variability_sweep_family_variance.png"),
        ("final_worst_config_loss_true", "Final worst-configuration loss", "variability_sweep_worst_config_loss.png"),
    ]
    for metric, ylabel, fname in specs:
        plt.figure(figsize=(8, 5))
        for (task_base, method), sub in run_df.groupby(["task_base", "method"]):
            agg = sub.groupby("variability_level")[metric].mean().sort_index()
            plt.plot(agg.index, agg.values, marker="o", label=f"{task_base} | {method}")
        plt.xlabel("Configuration variability level")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs configuration variability")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=160)
        plt.close()


def run_self_tests() -> None:
    exp = FamilyExperiment(n_qubits=3, depth=2)
    family = exp.build_family(0.01, 123)
    cfg = family[0]
    theta = np.linspace(-0.2, 0.2, exp.n_params)
    task = TaskSpec("tfim_test", tfim_hamiltonian_3q(0.7))

    psi = exp._statevector(theta, cfg)
    val = exp.term_expectation(psi, "ZZI")
    assert isinstance(val, float)
    assert -1.000001 <= val <= 1.000001

    G = exp.qgt_real(theta, cfg)
    assert G.shape == (exp.n_params, exp.n_params)
    assert np.all(np.isfinite(G))
    assert np.allclose(G, G.T, atol=1e-8)
    assert float(np.min(np.linalg.eigvalsh(G))) > -1e-5

    grad = exp.gradient_param_shift(theta, cfg, task)
    assert grad.shape == theta.shape
    assert np.all(np.isfinite(grad))

    losses = exp.family_losses(theta, family, task)
    assert losses.shape == (3,)
    assert np.all(np.isfinite(losses))

    runner = FAVORunner(exp, RunConfig(max_iters=3))
    run_df, hist_df = runner.run_task_seed(task, family, seed=0)
    assert not run_df.empty
    assert not hist_df.empty
    assert set(run_df["method"]) == {"GD", "QNG", "BlockQNG", "UF-GD", "UF-QNG-Mix"}
    print("Self-tests passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final Option A family-aware VQA experiment.")
    parser.add_argument("--output-dir", type=str, default="outputs_favo_final")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--variability-levels", type=float, nargs="+", default=[0.00, 0.01, 0.02, 0.05])
    parser.add_argument("--max-iters", type=int, default=60)
    parser.add_argument("--lr-gd", type=float, default=0.08)
    parser.add_argument("--lr-qng", type=float, default=0.08)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    if args.run_tests:
        run_self_tests()
        return

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    exp = FamilyExperiment(n_qubits=3, depth=2)
    run_cfg = RunConfig(lr_gd=args.lr_gd, lr_qng=args.lr_qng, lam=args.lam, max_iters=args.max_iters)
    runner = FAVORunner(exp, run_cfg)

    tasks_base = [
        TaskSpec("tfim_3q_d2", tfim_hamiltonian_3q(0.7)),
        TaskSpec("heis_3q_d2", heisenberg_hamiltonian_3q()),
    ]

    all_runs = []
    all_hist = []
    for variability_level in args.variability_levels:
        for tb in tasks_base:
            task = TaskSpec(f"{tb.name}__var_{variability_level:.3f}", tb.h_terms)
            for seed in args.seeds:
                family = exp.build_family(variability_level, seed + int(10000 * variability_level))
                run_df, hist_df = runner.run_task_seed(task, family, seed)
                run_df["variability_level"] = variability_level
                hist_df["variability_level"] = variability_level
                all_runs.append(run_df)
                all_hist.append(hist_df)
                print(f"Completed task={task.name}, seed={seed}")

    run_df = pd.concat(all_runs, ignore_index=True)
    hist_df = pd.concat(all_hist, ignore_index=True)

    summary_df = summarize_results(run_df)
    paired_qng_df = paired_differences(run_df, "QNG")
    paired_ufqng_df = paired_differences(run_df, "UF-QNG-Mix")

    run_df.to_csv(outdir / "run_level.csv", index=False)
    hist_df.to_csv(outdir / "iteration_history.csv", index=False)
    summary_df.to_csv(outdir / "summary_by_method.csv", index=False)
    paired_qng_df.to_csv(outdir / "paired_differences_vs_qng.csv", index=False)
    paired_ufqng_df.to_csv(outdir / "paired_differences_vs_uf_qng_mix.csv", index=False)

    run_df["task_base"] = run_df["task"].str.split("__var_").str[0]
    variability_summary = (
        run_df.groupby(["task_base", "variability_level", "method"])[
            ["final_family_loss_true", "final_family_var_true", "final_worst_config_loss_true"]
        ]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    variability_summary.columns = ["__".join([c for c in col if c]) for col in variability_summary.columns.to_flat_index()]
    variability_summary.to_csv(outdir / "variability_sweep_summary.csv", index=False)

    for task_name in sorted(hist_df["task"].unique()):
        plot_task_history(hist_df, task_name, outdir)
    plot_variability_sweep(run_df, outdir)

    meta = {
        "n_qubits": exp.n_qubits,
        "depth": exp.depth,
        "n_params": exp.n_params,
        "base_rzz": exp.base_rzz,
        "run_config": dataclasses.asdict(run_cfg),
        "seeds": args.seeds,
        "variability_levels": args.variability_levels,
        "methods": ["GD", "QNG", "BlockQNG", "UF-GD", "UF-QNG-Mix"],
        "tasks": [t.name for t in tasks_base],
        "paper_direction": "Option A: family-aware optimization under configuration variability",
    }
    with open(outdir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved outputs to {outdir.resolve()}")


if __name__ == "__main__":
    main()
