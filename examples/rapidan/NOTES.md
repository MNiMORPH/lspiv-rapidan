# Rapidan Dam — processing notes

Site: Rapidan Dam, Blue Earth River, Martin County, MN.
Dam failed June 23, 2024. Footage spans peak flood through post-failure
channel adjustment.

Collaborator: Zach Hilgendorf (MSU Mankato) — provides SfM orthophoto and DSM.
Results repo: https://github.com/MNiMORPH/rapidan-lspiv (LFS-enabled; see README there).

---

## Clip inventory — status as of 2026-06-15

11 clips in `data/raw/`. Full assessment of all 79 reviewed clips in
`data/inventory/assessment.json`; representative frames in
`data/inventory/frames/<clip>/frame_mid.jpg`.

| File | Date | Flow | Status | Notes |
|---|---|---|---|---|
| `MAX_0102.MP4` | 2024-11-15 | Base | **Done** | 2 s test clip; 177 SIFT inliers |
| `DJI_0586_RAPIDAN_nadir.MP4` | 2024-06-28 | Very high | On hold | 5 SIFT inliers |
| `DJI_0658_RAPIDAN_nadir.MP4` | 2024-06-30 | High/receding | On hold | Not attempted |
| `DJI_0672_RAPIDAN_nadir.MP4` | 2024-06-30 | High/receding | On hold | Not attempted; use upstream quadrant only |
| `DJI_0860_RAPIDAN_nadir.MP4` | 2024-07-09 | Mod-high | On hold | Not attempted |
| `MAX_0015_nadir.MP4` | 2024-08-06 | Low-moderate | On hold | 9 SIFT inliers — degenerate |
| `DJI_0022.MP4` | 2025-05-15 | Elevated | On hold | 4 SIFT inliers |
| `DJI_0023.MP4` | 2025-05-15 | Elevated | On hold | 4 SIFT inliers |
| `DJI_0024.MP4` | 2025-05-15 | Elevated | On hold | 4 SIFT inliers |
| `MAX_0321.MP4` | 2025-06-24 | Elevated | On hold | 5–6 SIFT inliers |
| `MAX_0322.MP4` | 2025-06-24 | Elevated | On hold | 5–6 SIFT inliers |

`_nadir` = trimmed subclip isolating near-nadir hover. `DJI_0022/023/024`
were `.MOV`, remuxed losslessly: `ffmpeg -i input.MOV -c copy output.MP4`.

---

## SIFT georeferencing: why most clips are on hold

The orthophoto (`data/orthophoto.tif`) was acquired ~November 2024 at base
flow — the same conditions as MAX_0102 (177 inliers). All other clips differ
in water level, camera type, or scene epoch, yielding 4–9 inliers and
degenerate homographies.

**What is needed to resume:**
1. Additional SfM orthophotos from Zach (one per flow condition)
2. Manual GCPs on stable features (rock ledge edges, concrete remnants)
3. GPS/IMU direct georeferencing from DJI flight logs

Andy emailed Zach 2026-06-15 to request additional orthophotos.

**Always inspect `results/<clip>/georeference_debug.png`** after georeferencing.
If inlier count is <20, treat the result as suspect.

---

## Key config values for this site

See `examples/rapidan/config.yaml`. Notable site-specific values:

| Parameter | Value | Notes |
|---|---|---|
| `piv.window_size` | `20` | ~1 m interrogation window at this scale |
| `piv.h_a` | `0.0` | Update with gauge data per clip date |
| `georeference_method` | `sift` | Uses Zach's orthophoto |
| `video_extension` | `MP4` | All clips are .MP4 (DJI .MOV were remuxed) |

---

## Sharing results

```bash
CLIP=MAX_0102
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
