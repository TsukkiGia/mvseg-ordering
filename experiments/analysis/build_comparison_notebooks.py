from __future__ import annotations

from pathlib import Path
import nbformat as nbf


def comparison_code_block(exp: str) -> str:
    # Build paths relative to the repository root to avoid notebook CWD issues.
    root_detect = (
        "def _repo_root():\n"
        "    p = Path().resolve()\n"
        "    for cand in [p] + list(p.parents):\n"
        "        if (cand / 'experiments' / 'scripts').is_dir():\n"
        "            return cand\n"
        "    return p\n\n"
        "ROOT = _repo_root()\n"
    )

    if exp == "2":
        path_init = (
            root_detect +
            "paths = {\n"
            "    'commit_label_90': ROOT / 'experiments' / 'scripts' / 'experiment_2_MM_commit_label_90',\n"
            "    'commit_label_97': ROOT / 'experiments' / 'scripts' / 'experiment_2_MM_commit_label_97',\n"
            "    'commit_pred_90': ROOT / 'experiments' / 'scripts' / 'experiment_2_MM_commit_pred_90',\n"
            "    'commit_pred_97': ROOT / 'experiments' / 'scripts' / 'experiment_2_MM_commit_pred_97',\n"
            "}\n"
        )
    else:
        path_init = (
            root_detect +
            f"base = ROOT / 'experiments' / 'scripts' / 'experiment_{exp}'\n"
            "paths = {\n"
            "    'commit_label_90': base / 'commit_label_90',\n"
            "    'commit_label_97': base / 'commit_label_97',\n"
            "    'commit_pred_90': base / 'commit_pred_90',\n"
            "    'commit_pred_97': base / 'commit_pred_97',\n"
            "}\n"
        )

    code = (
        "from pathlib import Path\n"
        "from IPython.display import display, HTML\n\n"
        + path_init +
        "\n"
        "def plan_a_figures(root: Path) -> Path: return root / 'A' / 'results' / 'figures'\n"
        "def plan_b_figures(root: Path) -> Path: return root / 'B' / 'figures'\n\n"
        "def list_pngs(p: Path):\n"
        "    if not p.exists():\n"
        "        return {}\n"
        "    return {f.name: f for f in sorted(p.glob('*.png'))}\n\n"
        "def build_pairs(dir90: Path, dir97: Path, filter_fn=lambda n: True):\n"
        "    d90, d97 = list_pngs(dir90), list_pngs(dir97)\n"
        "    commons = sorted(n for n in (d90.keys() & d97.keys()) if filter_fn(n))\n"
        "    return [(n, str(d90[n]), str(d97[n])) for n in commons]\n\n"
        "def build_pairs_map(dir90: Path, dir97: Path, filter_fn=lambda n: True):\n"
        "    d90, d97 = list_pngs(dir90), list_pngs(dir97)\n"
        "    names = sorted(n for n in (d90.keys() & d97.keys()) if filter_fn(n))\n"
        "    return {n: (str(d90[n]), str(d97[n])) for n in names}\n\n"
        "def show_interleaved(heading, label_map, pred_map, col_titles=('cutoff 0.90','cutoff 0.97')):\n"
        "    display(HTML(f'<h2 style=\"margin-top:18px\">{heading}</h2>'))\n"
        "    names = sorted(set(label_map.keys()) | set(pred_map.keys()))\n"
        "    if not names:\n"
        "        display(HTML('<p><em>No matching figures.</em></p>'))\n"
        "        return\n"
        "    rows = []\n"
        "    for name in names:\n"
        "        rows.append(f'<tr><td colspan=2 style=\"text-align:center;font-weight:600;padding-top:14px\">{name}</td></tr>')\n"
        "        if name in label_map:\n"
        "            p90, p97 = label_map[name]\n"
        "            rows.append(\n"
        "                f'<tr><td style=\\\"width:50%;text-align:center\\\"><div style=\\\"font-size:12px;color:#555\\\">commit_label — {col_titles[0]}</div><img src=\\\"{p90}\\\" style=\\\"max-width:100%;border:1px solid #ddd\\\"/></td>'\n"
        "                f'<td style=\\\"width:50%;text-align:center\\\"><div style=\\\"font-size:12px;color:#555\\\">commit_label — {col_titles[1]}</div><img src=\\\"{p97}\\\" style=\\\"max-width:100%;border:1px solid #ddd\\\"/></td></tr>'\n"
        "            )\n"
        "        if name in pred_map:\n"
        "            p90, p97 = pred_map[name]\n"
        "            rows.append(\n"
        "                f'<tr><td style=\\\"width:50%;text-align:center\\\"><div style=\\\"font-size:12px;color:#555\\\">commit_pred — {col_titles[0]}</div><img src=\\\"{p90}\\\" style=\\\"max-width:100%;border:1px solid #ddd\\\"/></td>'\n"
        "                f'<td style=\\\"width:50%;text-align:center\\\"><div style=\\\"font-size:12px;color:#555\\\">commit_pred — {col_titles[1]}</div><img src=\\\"{p97}\\\" style=\\\"max-width:100%;border:1px solid #ddd\\\"/></td></tr>'\n"
        "            )\n"
        "    html = \"<table style=\\\"width:100%\\\">{}</table>\".format(\"\\n\".join(rows))\n"
        "    display(HTML(html))\n\n"
        "is_plan_c = lambda n: n.startswith('eval_') or n.startswith('plan_c_')\n"
        "is_not_plan_c = lambda n: not is_plan_c(n)\n"
    )
    return code


def build_notebook(exp: str, out_path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(
        f"# Experiment {exp} — Figures Comparison\n\n"
        "Compare commit_label vs commit_pred at Dice cutoffs 0.90 and 0.97.\n"
        "Grouped by Plan A, Plan C, and Plan B."
    ))
    nb.cells.append(nbf.v4.new_code_cell(comparison_code_block(exp)))

    nb.cells.append(nbf.v4.new_markdown_cell("## Plan A (non‑Plan C) — Interleaved Label/Pred"))
    nb.cells.append(nbf.v4.new_code_cell(
        "label_map = build_pairs_map(plan_a_figures(paths['commit_label_90']), plan_a_figures(paths['commit_label_97']), is_not_plan_c)\n"
        "pred_map = build_pairs_map(plan_a_figures(paths['commit_pred_90']), plan_a_figures(paths['commit_pred_97']), is_not_plan_c)\n"
        "show_interleaved('Plan A — Interleaved (commit_label & commit_pred)', label_map, pred_map)\n"
    ))

    nb.cells.append(nbf.v4.new_markdown_cell("## Plan C (eval_* + plan_c_*) — Interleaved Label/Pred"))
    nb.cells.append(nbf.v4.new_code_cell(
        "label_map = build_pairs_map(plan_a_figures(paths['commit_label_90']), plan_a_figures(paths['commit_label_97']), is_plan_c)\n"
        "pred_map = build_pairs_map(plan_a_figures(paths['commit_pred_90']), plan_a_figures(paths['commit_pred_97']), is_plan_c)\n"
        "show_interleaved('Plan C — Interleaved (commit_label & commit_pred)', label_map, pred_map)\n"
    ))

    nb.cells.append(nbf.v4.new_markdown_cell("## Plan B — Interleaved Label/Pred"))
    nb.cells.append(nbf.v4.new_code_cell(
        "label_map = build_pairs_map(plan_b_figures(paths['commit_label_90']), plan_b_figures(paths['commit_label_97']))\n"
        "pred_map = build_pairs_map(plan_b_figures(paths['commit_pred_90']), plan_b_figures(paths['commit_pred_97']))\n"
        "show_interleaved('Plan B — Interleaved (commit_label & commit_pred)', label_map, pred_map)\n"
    ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out_path)


def main() -> None:
    root = Path('experiments/analysis')
    build_notebook('2', root / 'exp2_figures_comparison.ipynb')
    build_notebook('3', root / 'exp3_figures_comparison.ipynb')
    build_notebook('4', root / 'exp4_figures_comparison.ipynb')
    print('Notebooks written to', root)


if __name__ == '__main__':
    main()
