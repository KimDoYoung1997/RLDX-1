"""Turn a 20 fps LeRobot v2.1 split into a 10 fps one, as if it had been recorded at 10 fps.

Keeps the even frames of every episode (frame 0, 2, 4, ...), renumbers
frame_index / timestamp / index, re-encodes each video from the same even
frames at 10 fps, halves the subtask segment boundaries, recomputes the
per-episode stats and sets fps=10 in info.json. No stats.json is written
(the trainers generate it from the data, as for the 20 fps splits).

    python downsample_10hz.py SRC DST [workers] [phase]

phase 0 keeps frames 0, 2, 4, ... (default); phase 1 keeps 1, 3, 5, ... — the
two phases of one 20 fps episode are two 10 fps episodes offset by 50 ms.
"""
import json, math, os, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC, DST = sys.argv[1], sys.argv[2]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
PHASE = int(sys.argv[4]) if len(sys.argv) > 4 else 0
assert PHASE in (0, 1)
FPS_IN, FPS_OUT = 20, 10
VKEY = "observation.images.cam_head"


def rl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def new_idx(f):
    # number of kept frames (PHASE, PHASE+2, ...) strictly before original frame f
    return max(0, math.ceil((f - PHASE) / 2))


def half_segments(segs):
    # half-open [start, end) on the original frames -> the kept frames inside it
    return [{**s, "start": new_idx(s["start"]), "end": new_idx(s["end"])} for s in segs]


def nb_frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def do_episode(job):
    ei, n_src, index_offset = job
    pin = f"{SRC}/data/chunk-000/episode_{ei:06d}.parquet"
    t = pq.read_table(pin)
    assert t.num_rows == n_src, (ei, t.num_rows, n_src)
    fi = t.column("frame_index").to_numpy()
    assert (fi == np.arange(n_src)).all(), ei
    keep = np.arange(PHASE, n_src, 2)
    n = len(keep)
    t = t.take(pa.array(keep))
    def put(tab, name, arr):
        i = tab.schema.get_field_index(name)
        return tab.set_column(i, tab.schema.field(i), pa.array(arr, type=tab.schema.field(i).type))
    t = put(t, "frame_index", np.arange(n))
    t = put(t, "timestamp", (np.arange(n) / FPS_OUT).astype(np.float32))
    t = put(t, "index", index_offset + np.arange(n))
    pout = f"{DST}/data/chunk-000/episode_{ei:06d}.parquet"
    pq.write_table(t, pout)  # schema (incl. huggingface metadata) carried over from the source table

    vin = f"{SRC}/videos/chunk-000/{VKEY}/episode_{ei:06d}.mp4"
    vout = f"{DST}/videos/chunk-000/{VKEY}/episode_{ei:06d}.mp4"
    assert nb_frames(vin) == n_src, (ei, "src video frames != parquet rows")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vin,
                    "-vf", f"select='eq(mod(n,2),{PHASE})',setpts=N/(10*TB)",
                    "-r", str(FPS_OUT), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-g", "20", "-an", vout], check=True)
    nv = nb_frames(vout)
    assert nv == n, (ei, "dst video frames", nv, "!= rows", n)

    stats = {}
    for col in ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]:
        a = np.stack(t.column(col).to_numpy(zero_copy_only=False)).astype(np.float64)
        if a.ndim == 1:
            a = a[:, None]
        stats[col] = {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
                      "std": a.std(0).tolist(), "count": [n]}
    return ei, n, stats


def main():
    os.makedirs(f"{DST}/data/chunk-000", exist_ok=False)
    os.makedirs(f"{DST}/videos/chunk-000/{VKEY}")
    os.makedirs(f"{DST}/meta")
    eps = rl(f"{SRC}/meta/episodes.jsonl")
    jobs, off = [], 0
    for e in eps:
        n_new = len(range(PHASE, e["length"], 2))
        jobs.append((e["episode_index"], e["length"], off))
        off += n_new
    with ProcessPoolExecutor(WORKERS) as ex:
        res = {ei: (n, st) for ei, n, st in ex.map(do_episode, jobs)}

    with open(f"{DST}/meta/episodes.jsonl", "w") as f:
        for e in eps:
            e = dict(e); e["length"] = res[e["episode_index"]][0]
            if e.get("sub_tasks"):
                e["sub_tasks"] = half_segments(e["sub_tasks"])
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    old_stats = {d["episode_index"]: d for d in rl(f"{SRC}/meta/episodes_stats.jsonl")}
    with open(f"{DST}/meta/episodes_stats.jsonl", "w") as f:
        for e in eps:
            ei = e["episode_index"]; n, st = res[ei]
            d = old_stats[ei]["stats"]
            for k in d:  # video stats: same frames' distribution, just fewer of them
                if k not in st:
                    st[k] = {**d[k], "count": [n]}
            f.write(json.dumps({"episode_index": ei, "stats": st}) + "\n")
    if os.path.exists(f"{SRC}/meta/custom_meta.jsonl"):
        with open(f"{DST}/meta/custom_meta.jsonl", "w") as f:
            for c in rl(f"{SRC}/meta/custom_meta.jsonl"):
                if c.get("sub_tasks"):
                    c = {**c, "sub_tasks": half_segments(c["sub_tasks"])}
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    for fn in ["tasks.jsonl", "modality.json"]:
        shutil.copy2(f"{SRC}/meta/{fn}", f"{DST}/meta/{fn}")
    info = json.load(open(f"{SRC}/meta/info.json"))
    info["fps"] = FPS_OUT
    info["total_frames"] = sum(n for n, _ in res.values())
    info["features"][VKEY]["info"]["video.fps"] = FPS_OUT
    json.dump(info, open(f"{DST}/meta/info.json", "w"), indent=4)
    print(f"DONE {DST}: {len(eps)} ep, {info['total_frames']} frames (src {sum(e['length'] for e in eps)})")


if __name__ == "__main__":
    main()
