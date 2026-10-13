#!/usr/bin/env python3
"""Renders the benchmark CSV into PNG charts.

Usage: plot_results.py results/results.csv results/
"""
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"dag": "#1f77b4", "dag_nohc": "#ff7f0e", "torch": "#2ca02c", "tree": "#d62728"}
LABELS = {"dag": "Tensor DAG (hash-consing)", "dag_nohc": "Tensor DAG (no hash-consing)",
          "torch": "LibTorch Autograd", "tree": "Naive tree AST"}


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def plot_nodes_vs_depth(df, out_dir):
    """Symbolic size: backward nodes vs depth for the chain task (small n)."""
    sub = df[(df.task == "chain") & (df.suite.isin(["chain", "scaling"]))]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for be in ["dag", "dag_nohc", "tree"]:
        s = sub[(sub.backend == be)].groupby("depth").agg({"bwd_nodes": "max", "diff_us": "max", "tree_capped": "max"}).reset_index() \
            if be == "tree" else sub[(sub.backend == be)].groupby("depth").agg({"bwd_nodes": "max", "diff_us": "median"}).reset_index()
        if be == "tree":
            s = s[s.tree_capped == 0]
        if s.empty:
            continue
        axes[0].plot(s.depth, s.bwd_nodes, "o-", color=COLORS[be], label=LABELS[be])
        axes[1].plot(s.depth, s.diff_us, "o-", color=COLORS[be], label=LABELS[be])
    for ax, title in zip(axes, ["Gradient expression size (nodes)", "Differentiation time (us)"]):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("chain depth")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Chain task: tree AST vs Tensor DAG symbolic differentiation (tree points past the node cap omitted)")
    save(fig, out_dir, "fig_chain_nodes_time.png")


def plot_scaling(df, out_dir):
    """Linearity: differentiation time and backward nodes vs forward nodes."""
    sub = df[(df.suite == "scaling") & (df.backend == "dag")]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for task, marker in [("chain", "o"), ("diamond", "s"), ("mlp", "^")]:
        s = sub[sub.task == task].sort_values("fwd_nodes")
        if s.empty:
            continue
        axes[0].plot(s.fwd_nodes, s.bwd_nodes, marker + "-", label=task)
        axes[1].plot(s.fwd_nodes, s.diff_us, marker + "-", label=task)
    axes[0].set_title("backward nodes vs forward nodes")
    axes[1].set_title("symbolic VJP construction time (us) vs forward nodes")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("forward DAG nodes")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Tensor DAG symbolic differentiation scaling (hash-consing on)")
    save(fig, out_dir, "fig_scaling.png")


def plot_sharing(df, out_dir):
    sub = df[df.suite == "sharing"]
    if sub.empty:
        return
    ns = sorted(sub.n.unique())
    fig, axes = plt.subplots(1, len(ns) + 1, figsize=(5 * (len(ns) + 1), 4))
    ax = axes[0]
    for be in ["dag", "dag_nohc", "torch"]:
        s = sub[(sub.backend == be) & (sub.n == ns[0])].sort_values("sharing")
        ax.plot(s.sharing, s.aten_ops_full, "o-", color=COLORS[be], label=LABELS[be])
    ax.set_title("ATen operator invocations (fwd+bwd)")
    ax.set_xlabel("duplicate derivations per layer (k)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    for ax, n in zip(axes[1:], ns):
        for be in ["dag", "dag_nohc", "torch"]:
            s = sub[(sub.backend == be) & (sub.n == n)].sort_values("sharing")
            ax.errorbar(s.sharing, s.total_median_us, yerr=[s.total_median_us * 0, s.total_p90_us - s.total_median_us],
                        fmt="o-", color=COLORS[be], label=LABELS[be], capsize=3)
        ax.set_title(f"fwd+bwd time, n={n} (median, P90 whisker)")
        ax.set_xlabel("duplicate derivations per layer (k)")
        ax.set_ylabel("us")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Sharing task: effect of hash-consing on executed operators and time")
    save(fig, out_dir, "fig_sharing.png")


def plot_exec(df, out_dir, suite, xkey, title, fname):
    sub = df[(df.suite == suite) & (df.backend.isin(["dag", "torch"]))]
    if sub.empty:
        return
    sub = sub.copy()
    sub["cfg"] = sub.label.str.replace(suite, "", regex=False)
    cfgs = list(dict.fromkeys(sub.sort_values(xkey).cfg))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.38
    for i, be in enumerate(["dag", "torch"]):
        s = sub[sub.backend == be].set_index("cfg").reindex(cfgs)
        x = [j + (i - 0.5) * width for j in range(len(cfgs))]
        axes[0].bar(x, s.fwd_median_us, width, color=COLORS[be], label=LABELS[be] + " forward")
        axes[0].bar(x, s.bwd_median_us, width, bottom=s.fwd_median_us, color=COLORS[be], alpha=0.5,
                    label=LABELS[be] + " backward")
        axes[1].bar(x, s.peak_malloc_delta_bytes / 1024 / 1024, width, color=COLORS[be], label=LABELS[be])
    axes[0].set_title("median fwd+bwd time (us, stacked)")
    axes[0].set_yscale("log")
    axes[1].set_title("peak malloc delta during fwd+bwd (MiB)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(cfgs, rotation=30, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    save(fig, out_dir, fname)


def plot_speedup(df, out_dir):
    sub = df[df.backend.isin(["dag", "torch"]) & (df.suite != "warmup")]
    if sub.empty:
        return
    piv = sub.pivot_table(index=["suite", "label", "total_defined_bytes"], columns="backend",
                          values=["fwd_median_us", "bwd_median_us", "total_median_us"], aggfunc="first")
    piv = piv.dropna()
    bytes_ = piv.index.get_level_values("total_defined_bytes").astype(float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, lab, mk in [("fwd_median_us", "forward", "o"), ("bwd_median_us", "backward", "s"), ("total_median_us", "fwd+bwd", "^")]:
        ratio = piv[(key, "torch")] / piv[(key, "dag")]
        ax.scatter(bytes_ / 1024 / 1024, ratio, marker=mk, label=lab, alpha=0.7)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("logical bytes defined by the fwd+bwd program (MiB)")
    ax.set_ylabel("LibTorch Autograd time / DAG executor time")
    ax.set_title("Speed ratio vs problem size (>1: DAG executor faster)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    save(fig, out_dir, "fig_speedup_vs_size.png")


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    df = pd.read_csv(csv)
    for col in ["tree_capped"]:
        if col not in df:
            df[col] = 0
    df["tree_capped"] = df["tree_capped"].fillna(0)
    os.makedirs(out_dir, exist_ok=True)
    plot_nodes_vs_depth(df, out_dir)
    plot_scaling(df, out_dir)
    plot_sharing(df, out_dir)
    plot_exec(df, out_dir, "chain", "n", "Chain task: DAG executor vs LibTorch Autograd", "fig_exec_chain.png")
    plot_exec(df, out_dir, "mlp", "hidden", "MLP task: DAG executor vs LibTorch Autograd", "fig_exec_mlp.png")
    plot_exec(df, out_dir, "nn", "batch", "MatMul+Bias+ReLU+Sum: DAG executor vs LibTorch Autograd", "fig_exec_nn.png")
    plot_exec(df, out_dir, "diamond", "width", "Diamond task: DAG executor vs LibTorch Autograd", "fig_exec_diamond.png")
    plot_speedup(df, out_dir)


if __name__ == "__main__":
    main()
