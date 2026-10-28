"""Public Codabench 17739 Development leaderboard (no auth): top rows + our rows, with the S loss budget vs the leader.
    python3 runs/codabench_board.py [owner]"""
import json, subprocess, sys
url = "https://www.codabench.org/api/phases/29580/get_leaderboard/"; subs = []
while url:
    d = json.loads(subprocess.run(["curl", "-s", "--max-time", "60", url], capture_output=True, check=True).stdout)  # curl: urllib fails here
    subs += d["submissions"]; url = d.get("next")
    url = url and "https://www.codabench.org/api/" + url.split("/api/", 1)[1]          # next points at an unreachable host
me = sys.argv[1] if len(sys.argv) > 1 else "mohhakami"
K = ["d3d_S", "d3d_r2_psi", "d3d_r2_qb", "d3d_dlcfs", "d3d_consistency", "g_ratio", "mast_S"]
rows = [dict(id=s["id"], owner=s["owner"], **{c["column_key"]: float(c["score"]) for c in s["scores"]}) for s in subs]
rows.sort(key=lambda r: -r.get("d3d_S", 0))
print(f"{len(rows)} rows; rank owner id | " + " ".join(K))
for i, r in enumerate(rows):
    if i < 8 or r["owner"] == me:
        print(f"{i + 1:3d} {r['owner'][:14]:14s} {r['id']} | " + " ".join(f"{r.get(k, float('nan')):.4f}" for k in K))
best = rows[0]; ours = [r for r in rows if r["owner"] == me]
if ours:
    o = max(ours, key=lambda r: r["d3d_S"])
    print(f"gap to #1 {best['d3d_S'] - o['d3d_S']:+.4f}: psi {0.55 * (best['d3d_r2_psi'] - o['d3d_r2_psi']):+.5f} qb {0.15 * (best['d3d_r2_qb'] - o['d3d_r2_qb']):+.5f} "
          f"dlcfs {0.10 * (o['d3d_dlcfs'] - best['d3d_dlcfs']):+.5f} cons {0.20 * (best['d3d_consistency'] - o['d3d_consistency']):+.5f}")
