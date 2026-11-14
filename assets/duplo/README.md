# DUPLO brick meshes

These meshes come from the official [LDraw parts library](https://library.ldraw.org/).
The LDraw parts are licensed under CC BY 4.0. Give credit to LDraw.org when you
share these files.

| File | LDraw part | BrickLink part | Size (width × depth × height, with studs) |
|---|---|---|---|
| `duplo_2x2_3437` | 3437 | 3437 | 32.0 × 32.0 × 23.6 mm |
| `duplo_2x4_3011` | 3011 | 3011 | 64.0 × 32.0 × 23.6 mm |
| `duplo_1x2x2_76371` | 76371 (with bottom tube) | 4066 / 76371 | 32.0 × 16.0 × 42.8 mm |
| `duplo_2x3_curved_top_2302` | 2302b | 2302 | 32.0 × 48.0 × 23.6 mm |

## Units and axes

- The units are metres. The Z axis is up.
- The origin is at the bottom centre of each brick.
- The long axis of `2302` is Y. The 2x2 studs are at +Y and the curved end is at −Y.

## Limits

- LDraw uses nominal sizes. A real brick is approximately 0.2 mm smaller than
  the nominal size, because LEGO leaves a small gap between bricks.
- The meshes are hollow underneath and are not watertight. Thus the mass that
  MuJoCo calculates from the mesh is wrong (for example, 37 g for the 2x2).
  Weigh the real bricks and set `mass` on each geom.
- MuJoCo uses the convex hull of each mesh for collisions. All four shapes are
  close to convex, so the hull is a good collision shape.

## Make the files again

```bash
uv run --no-project --with numpy python -I ldraw2mesh.py <cache-dir> <out-dir> \
  3437=duplo_2x2_3437 3011=duplo_2x4_3011 \
  76371=duplo_1x2x2_76371 2302b=duplo_2x3_curved_top_2302
```

The script downloads the part files and their sub-files from library.ldraw.org.
