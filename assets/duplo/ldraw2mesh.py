"""Convert official LDraw parts to STL/OBJ in metres, Z up. Fetches sub-files from library.ldraw.org."""
import os, struct, sys, urllib.request, urllib.error
import numpy as np

BASE = "https://library.ldraw.org/library/official/"
CACHE = sys.argv[1]
OUT = sys.argv[2]
LDU = 0.0004  # metres

def fetch(name):
    name = name.strip().lower().replace("\\", "/")
    for d in ("parts/", "p/"):
        path = os.path.join(CACHE, d + name)
        if os.path.exists(path):
            return open(path, encoding="utf-8", errors="replace").read()
    for d in ("parts/", "p/") * 3:
        try:
            data = urllib.request.urlopen(urllib.request.Request(BASE + d + name, headers={"User-Agent": "curl/8.0"}), timeout=20).read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404: continue
            continue
        except Exception:
            import time; time.sleep(1); continue
        path = os.path.join(CACHE, d + name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(data)
        return data
    raise FileNotFoundError(name)

def walk(name, M, invert, tris):
    text = fetch(name)
    ccw = True; certified = False; invnext = False
    for line in text.splitlines():
        t = line.split()
        if not t: continue
        if t[0] == "0":
            if len(t) > 2 and t[1] == "BFC":
                w = t[2:]
                if "CERTIFY" in w: certified = True
                if "CW" in w: ccw = False
                if "CCW" in w: ccw = True
                if "INVERTNEXT" in w: invnext = True
            continue
        if t[0] == "1":
            v = list(map(float, t[2:14]))
            sub = np.eye(4); sub[:3, 3] = v[0:3]; sub[:3, :3] = np.array(v[3:12]).reshape(3, 3)
            sub_inv = invert ^ invnext ^ (np.linalg.det(sub[:3, :3]) < 0)
            walk(" ".join(t[14:]), M @ sub, sub_inv, tris)
            invnext = False
            continue
        if t[0] in ("3", "4"):
            n = int(t[0]); pts = np.array(list(map(float, t[2:2 + 3 * n]))).reshape(n, 3)
            pts = (M[:3, :3] @ pts.T).T + M[:3, 3]
            faces = [pts[[0, 1, 2]]] if n == 3 else [pts[[0, 1, 2]], pts[[0, 2, 3]]]
            flip = (not ccw) ^ invert
            for f in faces:
                tris.append(f[::-1] if flip else f)
        invnext = False if t[0] != "0" else invnext

def convert(part, label):
    tris = []
    walk(part + ".dat", np.eye(4), False, tris)
    T = np.array(tris) * LDU
    # LDraw: -Y up. MuJoCo: Z up. (x, y, z) -> (x, z, -y) is a proper rotation.
    T = np.stack([T[..., 0], T[..., 2], -T[..., 1]], axis=-1)
    lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
    T -= np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])  # centre in XY, bottom at z=0
    stl = os.path.join(OUT, label + ".stl")
    with open(stl, "wb") as f:
        f.write((b"LDraw " + part.encode() + b" CC BY 4.0 LDraw.org").ljust(80, b" ")[:80])
        f.write(struct.pack("<I", len(T)))
        for tri in T:
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); l = np.linalg.norm(n); n = n / l if l else n
            f.write(struct.pack("<12fH", *n, *tri.ravel(), 0))
    obj = os.path.join(OUT, label + ".obj")
    with open(obj, "w") as f:
        f.write(f"# LDraw part {part}.dat converted to metres, Z up. Source: library.ldraw.org (CC BY 4.0)\n")
        for tri in T:
            for p in tri: f.write("v %.6f %.6f %.6f\n" % tuple(p))
        for i in range(len(T)): f.write("f %d %d %d\n" % (3 * i + 1, 3 * i + 2, 3 * i + 3))
    size = (hi - lo)[[0, 2, 1]] * 1000
    print(f"{label:28s} {part:>6s}  {len(T):6d} triangles  width {size[0]:.1f} x depth {size[2]:.1f} x height {size[1]:.1f} mm (incl. studs)")

os.makedirs(OUT, exist_ok=True)
for arg in sys.argv[3:]:
    part, label = arg.split("=")
    convert(part, label)
