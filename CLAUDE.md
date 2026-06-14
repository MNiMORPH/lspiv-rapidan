# lspiv-rapidan pipeline — notes for Claude

## How to run

The Snakefile uses `glob_wildcards` — it **automatically processes every `.MP4`
in `data/raw/`** through all three stages (stabilize → georeference → PIV).
No per-clip config changes are needed to add a clip; just place the file in
`data/raw/` and run:

```bash
snakemake --cores 4
```

To run a single clip without touching the others:

```bash
snakemake --cores 4 results/MAX_0321/PIVquiverFrame.png
```

---

## Clips in data/raw/ — status as of 2026-06-14

11 clips ready. Listed best-first for LSPIV quality; run in this order if
doing one at a time so you can verify georeferencing on the easiest ones first.

| File | Date | Flow | Camera motion | Special handling |
|---|---|---|---|---|
| `MAX_0321.MP4` | 2025-06-24 | Elevated | stationary | **Start here.** Full channel, excellent eddy texture |
| `MAX_0322.MP4` | 2025-06-24 | Elevated | stationary | Same session as MAX_0321, slightly shifted |
| `DJI_0024.MP4` | 2025-05-15 | Elevated | stationary | Cascade left + pool right; mask cascade zone (left ~40% of frame) |
| `DJI_0023.MP4` | 2025-05-15 | Elevated | stationary | Same scene as 0024; large pool eddy visible |
| `DJI_0022.MP4` | 2025-05-15 | Elevated | stationary | Same scene as 0024; 12 s |
| `MAX_0015_nadir.MP4` | 2024-08-06 | Low-moderate | minor | Near-nadir, full 20 s; pool below cascade has good texture |
| `DJI_0860_RAPIDAN_nadir.MP4` | 2024-07-09 | Mod-high | significant→stationary | **Trimmed** last ~10 s of 18 s original; only near-nadir portion |
| `DJI_0672_RAPIDAN_nadir.MP4` | 2024-06-30 | High/receding | stationary | Full 79 s; frame dominated by dam structure — **use upstream quadrant only**, mask dam and breach |
| `DJI_0658_RAPIDAN_nadir.MP4` | 2024-06-30 | High/receding | significant→stationary | **Trimmed** 7–20 s; near-nadir of plunge pool gorge |
| `DJI_0586_RAPIDAN_nadir.MP4` | 2024-06-28 | Very high | significant→stationary | **Trimmed** first ~8 s; high-flow near-nadir |
| `MAX_0102.MP4` | 2024-11-15 | Base | stationary | Already processed; 2 s test clip, mostly cascade white water |

`_nadir` suffix = trimmed subclip isolating the near-nadir hover segment of a
longer moving shot. Full assessment of all 79 reviewed clips is in
`data/inventory/assessment.json`; representative frames in
`data/inventory/frames/<clip>/frame_mid.jpg`.

---

## Critical: SIFT georeferencing on post-failure scenes

The SIFT georeferencing step matches the stabilized video frame against
Zach's SfM **orthophoto** (`data/orthophoto.tif`). **If the orthophoto was
taken before the dam failure (Jun 23, 2024), the scene in post-failure clips
will differ dramatically** — the dam is breached, the channel has shifted, a
large sand bar is exposed, and much of the former impoundment is gone.

SIFT matching relies on stable visual features shared between the video frame
and the orthophoto. In post-failure clips, the remaining fixed features are:
- The bridge structure (intact in most clips)
- The surviving dam walkway/concrete
- The gorge walls
- The right-bank road/parking area

**Check `results/<clip>/georeference_debug.png` for every new clip.** If SIFT
finds no matches or produces an obviously wrong warp, the clip cannot be
georeferenced automatically — escalate to Andy, who can try:
1. Providing a post-failure orthophoto from Zach
2. Manual GCP entry via `lspiv-georeference --frame N` on a frame where fixed
   features are visible
3. Skipping the clip

The May/Jun 2025 clips (looking nearly straight down into the breach channel)
are the most likely to fail SIFT — very few distinctive fixed features are
visible at that scale. Try `georeference: frame_number: 0` to use the first
frame, which might show more of the surrounding context.

---

## .MOV → .MP4 remux note

`DJI_0022.MP4`, `DJI_0023.MP4`, `DJI_0024.MP4` were originally `.MOV` files.
They were losslessly remuxed with `ffmpeg -i input.MOV -c copy output.MP4`.
Do **not** re-rename without this remux — the container headers differ even
though the H.264 codec is the same.

---

## Per-clip masking

The pipeline does not currently have an automated masking step. For clips where
only part of the frame is water (cascade + pool clips, DJI_0672 upstream
quadrant), results outside the water area will be noise. Options:

1. **Post-hoc**: crop/mask the output GeoTIFFs in QGIS or Python after
   processing — easiest short-term
2. **pyORC native**: pyORC supports a `mask` parameter in the camera config
   JSON — ask Andy whether to implement this per-clip

---

## Sharing results

After each clip processes successfully, copy outputs to the shared results repo:

```bash
CLIP=MAX_0321   # set per clip
RESULTS_REPO=~/dataanalysis/rapidan-lspiv

mkdir -p $RESULTS_REPO/results/$CLIP
cp results/$CLIP/*.png results/$CLIP/*.tif results/$CLIP/*.nc \
   results/$CLIP/*.gpkg $RESULTS_REPO/results/$CLIP/
cp data/camera_configs/$CLIP.json $RESULTS_REPO/camera_configs/

cd $RESULTS_REPO
git add results/$CLIP/ camera_configs/$CLIP.json
git commit -m "Add $CLIP results"
git push
```

The repo is at **https://github.com/MNiMORPH/rapidan-lspiv** — Zach Hilgendorf
(MSU Mankato) is the intended collaborator. LFS is already configured for
`*.tif`, `*.nc`, `*.gpkg`, `*.png`, `*.jpg`.

---

## Config reference

Key `config/config.yaml` parameters to verify or update per-clip run:

| Parameter | Current value | Notes |
|---|---|---|
| `video_extension` | `MP4` | Correct — all clips are .MP4 |
| `georeference_method` | `sift` | Keep; uses Zach's orthophoto |
| `piv.h_a` | `0.0` | **Water surface elevation** — update with actual gauge data per clip date if available |
| `piv.window_size` | `20` | 1 m chip at current scale; may need tuning for close-up clips |
| `piv.water_elev_m` | *(blank)* | Auto-detected from DSM if blank; verify per clip |
| `piv.end_frame` | *(blank)* | Leave blank (use all frames) |
