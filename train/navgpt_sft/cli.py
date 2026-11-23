"""navgpt_sft command line: train, eval, merge, inspect, from-g1d-logs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .config import load_config


def _cmd_train(args) -> None:
    from .trainer import train

    cfg = load_config(args.config, tuple(args.set))
    print("final checkpoint:", train(cfg))


def _cmd_eval(args) -> None:
    from .trainer import evaluate_checkpoint

    cfg = load_config(args.config, tuple(args.set))
    metrics = evaluate_checkpoint(cfg, adapter=args.adapter, manifests=args.manifest or None)
    print(json.dumps(metrics, indent=2))


def _cmd_merge(args) -> None:
    from .export import merge

    cfg = load_config(args.config, tuple(args.set)) if args.config else None
    base = args.base or (cfg.model.checkpoint if cfg else None)
    if not base:
        raise SystemExit("--base or --config is required")
    info = merge(base, args.checkpoint, args.out, device=args.device)
    print("merged checkpoint:", args.out, info)


def _cmd_from_logs(args) -> None:
    from .g1d_logs import convert

    runs = []
    for root in args.logs:
        root = Path(root)
        runs += [root] if (root / "events.jsonl").exists() else [p for p in root.iterdir() if p.is_dir()]
    statuses = () if args.any_status else tuple(args.status or ["arrived"])
    counts = convert(runs, args.out, statuses=statuses, instruction=args.instruction)
    print(json.dumps(counts))


def _cmd_inspect(args) -> None:
    import numpy as np

    from .episodes import NavIndex, load_episodes, make_sample

    cfg = load_config(args.config, tuple(args.set))
    manifests = [(s.manifest, s.kind) for s in cfg.data.train] + [(m, "nav") for m in cfg.data.val]
    for manifest, kind in manifests:
        if kind != "nav":
            print("{}: vl source".format(manifest))
            continue
        episodes = load_episodes(manifest, cfg.nav.view_num)
        index = NavIndex.build(episodes, cfg.nav.keyframe_m, cfg.nav.keyframe_rad)
        missing = sum(1 for ep in episodes for f in ep.frames for p in f.images if not Path(p).exists())
        keys = [len(k) for k in index.keyframes]
        print("{}: {} episodes, {} samples (keyframes), keyframes/episode {:.1f} (max {}), missing images {}, {}".format(
            manifest, len(episodes), len(index.items), float(np.mean(keys)) if keys else 0.0,
            max(keys) if keys else 0, missing, index.class_counts()))
        if index.items and args.show:
            rng = np.random.default_rng(0)
            for item in rng.choice(len(index.items), size=min(args.show, len(index.items)), replace=False):
                s = make_sample(index, int(item), rng, cfg.nav.keyframe_m, cfg.nav.keyframe_rad, cfg.nav.max_sample_size)
                print("  item {} [{}] instruction: {}".format(item, index.classes[item], s.instruction))
                print("    history steps {} -> current {}".format(s.history_steps, s.history[-1].images[0]))
                print("    waypoints (m, m, deg): {}".format(
                    [[round(float(x), 2), round(float(y), 2), round(math.degrees(t), 1)] for x, y, t in s.waypoints]))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="navgpt_sft", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def with_config(p, required=True):
        p.add_argument("--config", required=required)
        p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                       help="override a config value, e.g. --set train.epochs=1")
        return p

    p = with_config(sub.add_parser("train", help="LoRA + action-head fine-tuning"))
    p.set_defaults(func=_cmd_train)

    p = with_config(sub.add_parser("eval", help="offline waypoint metrics on navigation manifests"))
    p.add_argument("--adapter", help="training checkpoint directory; omit to evaluate the base checkpoint")
    p.add_argument("--manifest", action="append", help="evaluate these manifests instead of data.val")
    p.set_defaults(func=_cmd_eval)

    p = with_config(sub.add_parser("merge", help="training checkpoint -> full NavGPT3 checkpoint"), required=False)
    p.add_argument("--checkpoint", required=True, help="training checkpoint directory (checkpoint-<step>)")
    p.add_argument("--out", required=True)
    p.add_argument("--base", help="base NavGPT3 checkpoint (default: model.checkpoint of --config)")
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=_cmd_merge)

    p = with_config(sub.add_parser("inspect", help="dataset statistics and example targets"))
    p.add_argument("--show", type=int, default=3, help="example samples per manifest")
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("from-g1d-logs", help="navgpt_g1d run logs -> episode manifest")
    p.add_argument("logs", nargs="+", help="run directories or their parent (e.g. deploy/g1d/logs)")
    p.add_argument("--out", required=True, help="manifest .jsonl to write")
    p.add_argument("--status", action="append", help="run outcome to keep (repeatable; default: arrived)")
    p.add_argument("--any-status", action="store_true", help="keep every run regardless of outcome")
    p.add_argument("--instruction", help="replace the logged instruction")
    p.set_defaults(func=_cmd_from_logs)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
