#!/usr/bin/env python3
"""Prints the Markdown summary tables used in REPORT.md from a results CSV."""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/results.csv")


def fmt_us(v):
    if pd.isna(v):
        return "-"
    return f"{v/1000:.2f} ms" if v >= 1000 else f"{v:.1f} us"


def fmt_b(v):
    if pd.isna(v):
        return "-"
    for u in ["B", "KiB", "MiB", "GiB"]:
        if v < 1024:
            return f"{v:.0f} {u}" if u == "B" else f"{v:.1f} {u}"
        v /= 1024
    return f"{v:.1f} TiB"


def table(rows, cols):
    out = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(f(r) for _, f in cols) + " |")
    return "\n".join(out)


def by(be, suite=None):
    s = df[df.backend == be]
    if suite:
        s = s[s.suite == suite]
    return s.set_index("label")


print("## T1 Tree AST vs Tensor DAG (symbolic size, differentiation time, memory)\n")
dag, tree = by("dag"), by("tree")
rows = []
for label in ["chain(n=1024,depth=4)", "chain(n=1024,depth=16)", "chain(n=1024,depth=64)", "chain(n=1024,depth=256)",
              "sharing(n=1024,depth=8,k=1)", "sharing(n=1024,depth=8,k=2)", "sharing(n=1024,depth=8,k=4)",
              "diamond(n=4096,depth=4,width=2)", "diamond(n=4096,depth=4,width=4)", "diamond(n=4096,depth=4,width=8)",
              "mlp(batch=4,in=8,hidden=8,out=8,layers=8)", "mlp(batch=4,in=8,hidden=8,out=8,layers=32)",
              "mlp(batch=4,in=8,hidden=8,out=8,layers=128)"]:
    d, t = dag.loc[label], tree.loc[label]
    rows.append((label, d, t))
capped = lambda t: " (cap hit)" if t.tree_capped else ""
print(table(rows, [
    ("configuration", lambda r: r[0]),
    ("DAG fwd nodes", lambda r: f"{int(r[1].fwd_nodes)}"),
    ("DAG bwd nodes", lambda r: f"{int(r[1].bwd_nodes)}"),
    ("DAG diff time", lambda r: fmt_us(r[1].diff_us)),
    ("DAG graph bytes", lambda r: fmt_b(r[1].graph_node_bytes)),
    ("tree fwd nodes", lambda r: f"{int(r[2].fwd_nodes)}" if not r[2].tree_capped or r[2].fwd_nodes else "cap hit"),
    ("tree grad nodes", lambda r: f"{int(r[2].bwd_nodes)}" if r[2].bwd_nodes else "cap hit"),
    ("tree diff time", lambda r: (fmt_us(r[2].diff_us) if r[2].diff_us else "-") + capped(r[2])),
    ("tree bytes", lambda r: fmt_b(r[2].graph_node_bytes) if r[2].bwd_nodes else "-"),
]))

print("\n## T2 Hash-consing ablation: sharing task (depth 8)\n")
hc, nohc, to = by("dag", "sharing"), by("dag_nohc", "sharing"), by("torch", "sharing")
rows = [(l, hc.loc[l], nohc.loc[l], to.loc[l]) for l in hc.index]
print(table(rows, [
    ("configuration", lambda r: r[0]),
    ("nodes HC / noHC", lambda r: f"{int(r[1].total_nodes)} / {int(r[2].total_nodes)}"),
    ("HC hits fwd+bwd", lambda r: f"{int(r[1].hashcons_hits_fwd)}+{int(r[1].hashcons_hits_bwd)}"),
    ("ATen ops HC / noHC / torch", lambda r: f"{int(r[1].aten_ops_full)} / {int(r[2].aten_ops_full)} / {int(r[3].aten_ops_full)}"),
    ("fwd+bwd median HC / noHC / torch", lambda r: f"{fmt_us(r[1].total_median_us)} / {fmt_us(r[2].total_median_us)} / {fmt_us(r[3].total_median_us)}"),
    ("peak live HC / noHC", lambda r: f"{fmt_b(r[1].peak_live_bytes)} / {fmt_b(r[2].peak_live_bytes)}"),
]))

print("\n## T3 Hash-consing on tasks without syntactic duplication\n")
hc, nohc = by("dag"), by("dag_nohc")
rows = [(l, hc.loc[l], nohc.loc[l]) for l in ["chain(n=1024,depth=256)", "diamond(n=4096,depth=16,width=8)",
                                               "nn(batch=256,in=512,out=256)", "mlp(batch=128,in=256,hidden=256,out=64,layers=8)",
                                               "chain(n=8,depth=4096)", "diamond(n=8,depth=1024,width=4)"]]
print(table(rows, [
    ("configuration", lambda r: r[0]),
    ("bwd nodes HC / noHC", lambda r: f"{int(r[1].bwd_nodes)} / {int(r[2].bwd_nodes)}"),
    ("HC hits bwd / AD requests", lambda r: f"{int(r[1].hashcons_hits_bwd)} / {int(r[1].ad_create_requests)}"),
    ("ATen ops HC / noHC", lambda r: f"{int(r[1].aten_ops_full)} / {int(r[2].aten_ops_full)}"),
    ("fwd+bwd median HC / noHC", lambda r: f"{fmt_us(r[1].total_median_us)} / {fmt_us(r[2].total_median_us)}"),
]))

print("\n## T4 Scaling of symbolic differentiation (hash-consing on, medians of 5 fresh graphs)\n")
s = df[(df.suite == "scaling") & (df.backend == "dag")]
rows = [r for _, r in s.iterrows()]
print(table(rows, [
    ("configuration", lambda r: r.label),
    ("fwd nodes", lambda r: f"{int(r.fwd_nodes)}"),
    ("fwd edges", lambda r: f"{int(r.fwd_edges)}"),
    ("bwd nodes", lambda r: f"{int(r.bwd_nodes)}"),
    ("bwd/fwd", lambda r: f"{r.bwd_nodes/r.fwd_nodes:.2f}"),
    ("accumulations", lambda r: f"{int(r.ad_accumulations)}"),
    ("build", lambda r: fmt_us(r.build_us)),
    ("diff", lambda r: fmt_us(r.diff_us)),
    ("diff ns / fwd node", lambda r: f"{r.diff_us*1000/r.fwd_nodes:.0f}"),
    ("compile", lambda r: fmt_us(r.compile_us)),
]))

print("\n## T5 Execution: DAG executor vs LibTorch Autograd (median; ratio = torch / DAG)\n")
d, t = by("dag"), by("torch")
rows = [(l, d.loc[l], t.loc[l]) for l in d.index if d.loc[l].suite != "warmup"]
print(table(rows, [
    ("configuration", lambda r: r[0]),
    ("ops DAG/torch", lambda r: f"{int(r[1].aten_ops_full)}/{int(r[2].aten_ops_full)}"),
    ("fwd DAG", lambda r: fmt_us(r[1].fwd_median_us)),
    ("fwd torch", lambda r: fmt_us(r[2].fwd_median_us)),
    ("bwd DAG", lambda r: fmt_us(r[1].bwd_median_us)),
    ("bwd torch", lambda r: fmt_us(r[2].bwd_median_us)),
    ("total DAG (P90)", lambda r: f"{fmt_us(r[1].total_median_us)} ({fmt_us(r[1].total_p90_us)})"),
    ("total torch (P90)", lambda r: f"{fmt_us(r[2].total_median_us)} ({fmt_us(r[2].total_p90_us)})"),
    ("fwd ratio", lambda r: f"{r[2].fwd_median_us/r[1].fwd_median_us:.2f}"),
    ("bwd ratio", lambda r: f"{r[2].bwd_median_us/r[1].bwd_median_us:.2f}"),
    ("total ratio", lambda r: f"{r[2].total_median_us/r[1].total_median_us:.2f}"),
    ("fwd no-grad DAG/torch", lambda r: f"{fmt_us(r[1].fwd_only_median_us)}/{fmt_us(r[2].fwd_only_median_us)}"),
    ("peak live DAG", lambda r: fmt_b(r[1].peak_live_bytes)),
    ("peak malloc DAG/torch", lambda r: f"{fmt_b(r[1].peak_malloc_delta_bytes)}/{fmt_b(r[2].peak_malloc_delta_bytes)}"),
    ("grad max rel err", lambda r: f"{r[1].grad_max_rel_err:.1e}"),
]))

print("\n## T6 One-time costs vs steady execution (DAG) and first execution\n")
rows = [(l, d.loc[l], t.loc[l]) for l in ["chain(n=1024,depth=16)", "chain(n=65536,depth=256)", "chain(n=8,depth=4096)",
                                          "diamond(n=8,depth=1024,width=4)", "mlp(batch=128,in=1024,hidden=1024,out=64,layers=8)",
                                          "nn(batch=1024,in=1024,out=1024)"]]
print(table(rows, [
    ("configuration", lambda r: r[0]),
    ("build (cold)", lambda r: f"{fmt_us(r[1].build_us)} ({fmt_us(r[1].build_first_us)})"),
    ("diff (cold)", lambda r: f"{fmt_us(r[1].diff_us)} ({fmt_us(r[1].diff_first_us)})"),
    ("compile", lambda r: fmt_us(r[1].compile_us)),
    ("first exec DAG", lambda r: fmt_us(r[1].first_exec_us)),
    ("steady exec DAG", lambda r: fmt_us(r[1].total_median_us)),
    ("first exec torch", lambda r: fmt_us(r[2].first_exec_us)),
    ("steady torch", lambda r: fmt_us(r[2].total_median_us)),
    ("liveness: peak live / all defined", lambda r: f"{fmt_b(r[1].peak_live_bytes)} / {fmt_b(r[1].total_defined_bytes)}"),
]))
