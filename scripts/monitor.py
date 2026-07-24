#!/usr/bin/env python
"""Gradio dashboard for monitoring Hindi TTS training.

Launch:
  python scripts/monitor.py --log-dir outputs/logs/hindi_train_<timestamp>

Then open http://localhost:7860 in a browser.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_metrics(log_dir: str) -> list[dict]:
    path = Path(log_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_plot(records: list[dict]) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Total Loss", "Backbone & Depth Loss"))
    steps = [r["step"] for r in records]
    if not steps:
        return fig
    fig.add_trace(go.Scatter(x=steps, y=[r["loss"] for r in records], mode="lines", name="loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=[r.get("loss_backbone", 0) for r in records], mode="lines", name="backbone"), row=2, col=1)
    fig.add_trace(go.Scatter(x=steps, y=[r.get("loss_depth", 0) for r in records], mode="lines", name="depth"), row=2, col=1)
    fig.update_layout(height=600, title_text="Training Metrics")
    return fig


def create_dashboard(log_dir: str):
    def refresh():
        records = load_metrics(log_dir)
        fig = build_plot(records)
        latest = records[-1] if records else {}
        return fig, json.dumps(latest, indent=2) if latest else "No data yet"

    with gr.Blocks(title="Hindi TTS Training Monitor") as demo:
        gr.Markdown(f"## Hindi TTS Training Monitor\nLog dir: `{log_dir}`")
        plot = gr.Plot()
        stats = gr.Textbox(label="Latest step", lines=4)
        refresh_btn = gr.Button("Refresh")
        demo.load(refresh, outputs=[plot, stats])
        refresh_btn.click(refresh, outputs=[plot, stats])
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=None, help="path to log dir with metrics.jsonl")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.log_dir is None:
        log_dirs = sorted(Path("outputs/logs").glob("hindi_train_*"))
        if not log_dirs:
            print("No training logs found. Run scripts/train.py first.")
            sys.exit(1)
        log_dir = str(log_dirs[-1])
    else:
        log_dir = args.log_dir

    print(f"Monitoring: {log_dir}")
    print(f"Dashboard: http://localhost:{args.port}")
    demo = create_dashboard(log_dir)
    demo.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
