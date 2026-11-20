"""Complete a room's objects offline, with its perception (2026-11-01, the fast room mode): the SAM 3D meshes that
codex_agent.run_codex would otherwise make before the agent starts (objects3d.prepare_meshes) are made here into a copy
of the room's given evidence, so a codify run on that copy starts its agent at once.

usage: python -m codify.prepare_room <perception run> <given evidence> <out evidence> [--gpu 6] [--workers 2]
The perception run gives the decoded frames; the out directory is the given evidence plus assets/ (one folder per
completed object and default.json, the ids whose mesh agrees with its box), with its own manifest."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from codify import contract, objects3d


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", help="the perception run (work/decode holds the frames)")
    p.add_argument("given", help="the room's given evidence (codify/instances.py --evidence)")
    p.add_argument("out", help="the evidence with the meshes, written here")
    p.add_argument("--gpu", default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--machine", default=str(Path(__file__).resolve().parents[1] / "configs" / "machine.json"),
                   help="the machine config naming the SAM 3D environment and checkpoints")
    a = p.parse_args()
    run, given, out = Path(a.run).resolve(), Path(a.given).resolve(), Path(a.out).resolve()
    manifest = contract.verify_evidence(given)
    if out.exists():
        raise SystemExit(f"{out} exists")
    temp = out.with_name(out.name + ".tmp")
    shutil.rmtree(temp, ignore_errors=True)
    shutil.copytree(given, temp, ignore=shutil.ignore_patterns("manifest.json"))
    machine = json.loads(Path(a.machine).read_text())
    cfg = {**machine["objects3d"], "workers": a.workers, **({"gpu": a.gpu} if a.gpu is not None else {})}
    ws = SimpleNamespace(evidence=temp, run=temp.parent / (out.name + "_work"), decode_dir=run / "work" / "decode",
                         video=contract.read_json(run / "work" / "decode" / "video_info.json"),
                         perception=SimpleNamespace(parts={}, work=run / "work", config={}))
    ws.run.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record = objects3d.prepare_meshes(ws, cfg, log=lambda *x: print(*x, flush=True))
    contract.write_manifest(temp, manifest["mode"], {**manifest["sources"], "given": str(given),
                                                     "given_digest": manifest["digest"],
                                                     "meshes": "codify.prepare_room (objects3d.prepare_meshes)"})
    temp.rename(out)
    print(json.dumps({"evidence": str(out), "meshes_agree": len(record["ids"]), "tried": record["tried"],
                      "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
