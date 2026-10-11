"""Concatenate LeRobot v2.1 datasets (same fps, same state/action layout) into one.

    python concat_lerobot.py SPEC.json OUT_DIR [workers]

SPEC: {"reference": <dataset whose info.json / modality.json / parquet schema
the output follows>, "sources": [{"path", "label", "episodes": [...] | null,
"drop_columns": [...], "reencode": bool, "phase": int | null}, ...]}

Every output episode is renumbered (episode_index, index); frame_index /
timestamp are kept. Columns not in the reference schema must be listed in
drop_columns (e.g. 0826's observation.tactile) and are removed; list types are
cast to the reference schema. Videos are copied, or re-encoded to h264 from
the same frames when "reencode" is set (frame counts asserted either way).
meta/source_episodes.jsonl maps every output episode back to its source.
"""
import json, os, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VKEY = "observation.images.cam_head"


def rl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def nb_frames(path):
    return int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                               "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
                              capture_output=True, text=True, check=True).stdout.strip())


def do_episode(job):
    src, sei, oei, off, drop, reencode, schema_bytes, out, fps = job
    schema = pa.ipc.read_schema(pa.py_buffer(schema_bytes))
    t = pq.read_table(f"{src}/data/chunk-000/episode_{sei:06d}.parquet")
    t = t.drop_columns([c for c in drop if c in t.column_names])
    n = t.num_rows
    cols = {}
    for f in schema:
        if f.name == "episode_index":
            cols[f.name] = pa.array(np.full(n, oei), type=f.type)
        elif f.name == "index":
            cols[f.name] = pa.array(off + np.arange(n), type=f.type)
        else:
            cols[f.name] = t.column(f.name).cast(f.type)
    t2 = pa.table(cols, schema=schema)
    pq.write_table(t2, f"{out}/data/chunk-000/episode_{oei:06d}.parquet")
    vin = f"{src}/videos/chunk-000/{VKEY}/episode_{sei:06d}.mp4"
    vout = f"{out}/videos/chunk-000/{VKEY}/episode_{oei:06d}.mp4"
    if reencode:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vin, "-r", str(fps), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", "-g", "20", "-an", vout], check=True)
    else:
        shutil.copy2(vin, vout)
    nv = nb_frames(vout)
    assert nv == n, (src, sei, "video frames", nv, "!= rows", n)
    stats = {}
    for col in ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]:
        a = np.stack(t2.column(col).to_numpy(zero_copy_only=False)).astype(np.float64)
        a = a[:, None] if a.ndim == 1 else a
        stats[col] = {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
                      "std": a.std(0).tolist(), "count": [n]}
    return oei, n, stats


def main():
    spec = json.load(open(sys.argv[1])); out = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    ref = spec["reference"]
    info = json.load(open(f"{ref}/meta/info.json"))
    fps = info["fps"]
    ref_pq = sorted(os.listdir(f"{ref}/data/chunk-000"))[0]
    schema = pq.read_schema(f"{ref}/data/chunk-000/{ref_pq}")
    schema_bytes = schema.serialize().to_pybytes()
    task = rl(f"{ref}/meta/tasks.jsonl")
    assert len(task) == 1
    os.makedirs(f"{out}/data/chunk-000"); os.makedirs(f"{out}/videos/chunk-000/{VKEY}"); os.makedirs(f"{out}/meta")

    jobs, rows, off, oei = [], [], 0, 0
    for s in spec["sources"]:
        p = s["path"]
        si = json.load(open(f"{p}/meta/info.json"))
        assert si["fps"] == fps, (p, si["fps"], fps)
        assert rl(f"{p}/meta/tasks.jsonl")[0]["task"] == task[0]["task"], p
        eps = {e["episode_index"]: e for e in rl(f"{p}/meta/episodes.jsonl")}
        cm = {c["episode_index"]: c for c in rl(f"{p}/meta/custom_meta.jsonl")}
        vstats = {d["episode_index"]: d["stats"] for d in rl(f"{p}/meta/episodes_stats.jsonl")}
        keep = s.get("episodes") or sorted(eps)
        for sei in keep:
            e = eps[sei]
            jobs.append((p, sei, oei, off, s.get("drop_columns", []), s.get("reencode", False), schema_bytes, out, fps))
            rows.append({"oei": oei, "src": s, "sei": sei, "ep": e, "cm": cm.get(sei, {}),
                         "vstats": {k: v for k, v in vstats.get(sei, {}).items() if k.startswith("observation.images")}})
            off += e["length"]; oei += 1

    with ProcessPoolExecutor(workers) as ex:
        res = {o: (n, st) for o, n, st in ex.map(do_episode, jobs)}

    with open(f"{out}/meta/episodes.jsonl", "w") as fe, open(f"{out}/meta/custom_meta.jsonl", "w") as fc, \
         open(f"{out}/meta/episodes_stats.jsonl", "w") as fs, open(f"{out}/meta/source_episodes.jsonl", "w") as fm:
        for r in rows:
            o, e, s = r["oei"], r["ep"], r["src"]
            n, st = res[o]
            assert n == e["length"], (o, n, e["length"])
            ep = {"episode_index": o, "tasks": [task[0]["task"]], "length": n}
            if e.get("sub_tasks"):
                ep["sub_tasks"] = e["sub_tasks"]
            fe.write(json.dumps(ep, ensure_ascii=False) + "\n")
            src_name = r["cm"].get("source_dataset") or s["label"]
            c = {k: v for k, v in r["cm"].items() if k not in ("episode_index",)}
            c.update({"episode_index": o, "source_dataset": src_name})
            fc.write(json.dumps(c, ensure_ascii=False) + "\n")
            for k, v in r["vstats"].items():
                st[k] = {**v, "count": [n]}
            fs.write(json.dumps({"episode_index": o, "stats": st}) + "\n")
            fm.write(json.dumps({"episode_index": o, "source": s["label"], "source_episode_index": r["sei"],
                                 "phase": s.get("phase"), "length": n}) + "\n")
    shutil.copy2(f"{ref}/meta/tasks.jsonl", f"{out}/meta/tasks.jsonl")
    shutil.copy2(f"{ref}/meta/modality.json", f"{out}/meta/modality.json")
    info["total_episodes"] = len(rows)
    info["total_frames"] = off
    info["total_videos"] = len(rows)
    info["splits"] = {"train": f"0:{len(rows)}"}
    info["features"][VKEY]["info"]["video.codec"] = "h264"
    json.dump(info, open(f"{out}/meta/info.json", "w"), indent=4)
    print(f"DONE {out}: {len(rows)} ep, {off} frames")


if __name__ == "__main__":
    main()
