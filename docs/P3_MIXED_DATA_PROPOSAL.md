# P3 Mixed Camera-Conditioning Data Proposal

*Prepared 2026-07-13 for the Unsplash Gemma-caption dataset and the local
`PicturesTraining` collection.*

## Objective

Train an additive camera-conditioning branch without cropping images, changing
captions, or allowing records without metadata to alter the P1 baseline. Use
the large Unsplash set for scene and text diversity, and use the smaller local
set as a high-confidence source of camera geometry labels.

This is initially an association and conditional-control experiment, not a
claim that camera metadata alone improves unconditional image quality or that
observational photographs isolate physical FOV control.

## Measured Data

### Unsplash Gemma captions

The production Parquet contains 9,293 records:

| Field | Usable records |
|---|---:|
| Long image captions | 9,293 |
| Raw focal length | 7,907 |
| Aperture | 7,886 |
| ISO | 8,062 |
| Valid camera make/model | 8,219 |
| Trusted vertical FOV | 0 |

Strengths are caption and scene diversity. The limitation is that raw focal
length is not comparable across sensor sizes. Camera make/model can support an
offline sensor lookup, but possible post-capture crops lower the confidence of
derived physical FOV.

### PicturesTraining

The local directory contains 1,226 images and no caption sidecars. A complete
EXIF/XMP scan found:

| Field | Usable records |
|---|---:|
| Any EXIF | 1,226 |
| Focal length, f-number, ISO, exposure | 1,225 |
| Focal-plane X/Y resolution and unit | 1,225 |
| Matching EXIF pixel dimensions for strict provenance | 322 |
| Lens model/specification | 1,220 |
| Direct 35 mm-equivalent focal length | 16 |
| XMP explicitly reports no crop | 1,225 |
| Strict derived sensor diagonal within 10% of camera table | 288 |

The original exploratory audit reported 1,191 plausible crop factors by using
the decoded raster dimensions. The committed v2 strict audit does not treat those
dimensions as interchangeable with the EXIF dimensions associated with the
focal-plane resolution: only 322 records prove the relationship, and 288 satisfy
the record-level conjunction of dimension match, no XMP crop, focal presence,
and sensor-diagonal agreement. The remaining records need an ExifTool-based
provenance recovery, native-dimension evidence, or rejection from the physical
geometry subset. The distribution is also narrow: 795 images are from a Canon
EOS 100D, and the collection contains only five camera models. This dataset
must not dominate the mixed sampler.

## Unified Manifest

Build a Parquet manifest for each source with the same schema. Do not read EXIF
or search camera specifications in the training hot path.

Required fields:

```text
local_image_path
caption_short
caption_medium
caption_detailed
caption_camera_neutral
caption_full
image_width
image_height

vertical_fov_deg
horizontal_fov_deg
focal_length_mm
focal_length_35mm_equivalent
sensor_width_mm
sensor_height_mm
aperture_f_number
iso
exposure_time_seconds
flash_fired
white_balance
capture_type

camera_make
camera_model
lens_model
geometry_source
geometry_confidence
dimension_provenance
has_post_crop
source_dataset
split_group
```

`geometry_source` should be one of:

```text
direct_exif_35mm
focal_plane_resolution
camera_sensor_table
pseudo_fov
raw_focal_only
unknown
```

Keep a presence bit for every numeric conditioning field. Missing data must not
be converted into a physical zero.

## Preprocessing

### Local collection

1. Generate short, medium and long captions with the same captioning process as
   Unsplash. Retain paired `caption_camera_neutral` and `caption_full` variants.
   The neutral variant must exclude camera/lens model names, focal lengths,
   f-stops, ISO and technical lens terms such as wide-angle, telephoto, fisheye,
   bokeh and depth of field. Save the leakage scan result in the audit artifact.
2. Preserve every image unchanged. Continue max-edge downscaling and full-frame
   letterboxing in the loader; do not crop.
3. Prefer direct `FocalLengthIn35mmFilm` when present.
4. Otherwise derive the physical image-plane dimensions only after recording
   decoded dimensions, root `ImageWidth`/`ImageLength`, EXIF
   `PixelXDimension`/`PixelYDimension`, orientation, focal-plane X/Y resolution
   and unit, XMP crop bounds, and known native camera dimensions. Reject an
   unexplained mismatch; `HasCrop=False` alone does not prove an export was not
   resized. Then use:

   ```text
   physical_width_mm = pixel_width / FocalPlaneXResolution * unit_mm
   physical_height_mm = pixel_height / FocalPlaneYResolution * unit_mm
   ```

   Apply EXIF orientation before assigning horizontal and vertical axes. Decode
   focal-plane resolution-unit codes explicitly; an unknown unit invalidates
   the derivation.
5. Calculate direct image-axis FOV:

   ```text
   fov = 2 * atan(physical_dimension_mm / (2 * focal_length_mm))
   ```

6. Validate the derived sensor diagonal against a small versioned table for the
   five observed camera models. Accept within 10%; use the table or mark unknown
   for outliers.
   When only 35 mm-equivalent focal length is available, reconstruct axis sizes
   from the 35 mm diagonal (43.27 mm) and the output aspect ratio before deriving
   horizontal and vertical FOV; do not assume every image is 3:2.
7. Use XMP crop bounds as a confidence gate. `HasCrop=False` is high confidence;
   missing crop metadata is medium confidence; explicit crop requires either an
   active-area correction or exclusion from the geometry subset.
8. Group bursts and near-duplicates by capture time plus perceptual similarity.
   Put an entire group into one split to prevent train/validation leakage.

### Unsplash collection

1. Merge `upstream_json`, `json`, and `metadata` instead of stopping at the first
   nonempty container.
2. Use direct 35 mm-equivalent values when available.
3. Build a versioned `(camera_make, camera_model) -> sensor dimensions` table,
   beginning with the 100 most common models. Those models cover about 82% of
   the valid make/model records.
4. Derive FOV only when focal length and sensor geometry are both available.
   Mark these labels medium confidence because an unknown post-capture crop may
   narrow the actual FOV.
5. Treat raw focal length without sensor geometry as an auxiliary correlation
   feature, never as a physical FOV label. Drop it after trusted FOV coverage is
   sufficient, or place it in a separate experimental residual head; never mix
   it into the trusted geometry head.
6. Add pseudo-FOV labels only as a separate ablation. Do not silently mix them
   with direct or sensor-derived labels.

## Conditioning Model

Before mixed training, change every metadata group to a permanently centered
residual:

```text
residual(value) = head(value) - head(unknown)
```

The unknown value must produce exactly zero after every optimizer update. This
keeps metadata-free generation numerically on the P1 path.

Use separate additive heads so dataset-specific missingness cannot become an
easy source classifier:

```text
geometry head: manifest-provided vertical FOV
raw-focal head: raw focal length, experimental correlation baseline only
exposure head: aperture, ISO, exposure time, flash
capture head: photo/render/artwork (disabled until non-photo classes exist)
```

Each missing group independently contributes zero. Sum the three residuals into
the UNet timestep embedding. Keep raw physical focal length out of the primary
geometry head once sufficient FOV coverage exists.

Implementation status (2026-07-13): schema v3 implements independent permanently
centered trusted-FOV, raw-focal, exposure and optional capture heads. Capture is
disabled by default. Exact schema validation, metadata-container merging,
guided counterfactual diagnostics, phase-specific dropout, and explicit camera
experiment modes are implemented. Existing v1/v2 camera checkpoints are
intentionally incompatible. The offline manifest builder remains the next P3a
gate.

With an exactly centered unknown residual and a frozen connector/UNet, complete
metadata dropout produces zero camera gradient. Therefore set whole-record
camera dropout to zero for the camera-only phase. Natural per-field missingness
already trains partial conditions. Reintroduce record dropout only in a phase
that also has other trainable parameters.

## Training Curriculum

### P3a: association/memorization falsifier

- Warm-start the completed P1 connector.
- Freeze Gemma, connector, VAE and UNet.
- Train only a small geometry head on the high-confidence local subset.
- Use early stopping and a held-out scene-group split; 288 strict records are
  not sufficient evidence for production generalization.
- Train real-label, within-camera shuffled-label, and constant-condition runs
  with identical head capacity and budgets.
- Hold out an entire camera or camera+lens configuration in addition to scene
  grouping. Compare real versus null controls on held-out diffusion loss and
  guided-control metrics.
- Purpose: determine whether labels contain usable held-out signal while unknown
  metadata remains exactly P1. This stage cannot prove isolated physical FOV.

### P3b: mixed metadata training

Start only after Unsplash sensor lookup or pseudo-labeling provides useful FOV
coverage.

- Define runs by optimizer steps, not ambiguous source epochs. For the first
  9,293-record Unsplash pass, 80/20 requires about 2,323 local exposures.
  Draw deterministically from all 1,226 local records with a hard maximum of
  two exposures each. Mask the geometry loss outside the strict eligible subset;
  those records can still train supported exposure fields.
- Materialize a versioned offline mixed manifest for each run. Sample Unsplash
  without replacement once; sample local records deterministically with a hard
  two-exposure cap. Stop the run when either the optimizer-step budget or the
  manifest is exhausted; do not silently switch to pure Unsplash.
- Apply hierarchical probabilities/caps in this order: source, camera or
  camera+lens configuration, then FOV bin.
- Preserve source proportions inside camera-model and focal-length bins where
  possible.
- Train geometry only on direct/high- and medium-confidence FOV labels.
- Train the exposure head on both datasets wherever individual fields exist.
- Do not include fully unknown records merely to satisfy dropout; they have no
  trainable effect in a centered camera-only stage.

The current round-robin source loader is not sufficient for this policy because
it alternates sources equally until one is exhausted. Make the loader consume
the offline mixed manifest before P3b; a runtime weighted sampler is optional,
not required for the first reproducible experiment.

### P3c: controlled physical validation

Observational cross-camera holdout does not remove the confounding between FOV,
camera-to-subject distance, scene category and photographer composition. Before
claiming physical control, add paired same-scene focal sweeps with fixed or
measured camera pose, or controlled synthetic renders with depth/pose labels.
Evaluate fixed-content prompts and projection/framing statistics. Perspective is
primarily determined by camera position; do not describe a focal sweep as direct
perspective control.

### P3d: optional joint refinement

Only if P3b/P3c show useful control without short-prompt regression:

- Unfreeze the connector at a substantially lower learning rate, or run the
  separately gated SaRA phase.
- Reintroduce a small metadata-drop probability so metadata-free examples train
  the newly unfrozen path.
- Keep the camera unknown residual algebraically fixed at zero.

## Evaluation

Compare four checkpoints:

```text
P1 baseline
P3a local-only geometry
P3b mixed high-confidence geometry
P3b plus pseudo-FOV, if attempted
P3a shuffled-label control
P3a constant-condition control
P3c paired or synthetic controlled validation
```

Required tests:

1. **Unknown identity:** after optimization, unknown camera conditioning must
   produce exactly the P1 camera residual of zero.
2. **Guided counterfactual:** compare actual guided noise predictions for FOV A
   and B. Keep CFG-delta difference as a secondary text-camera interaction
   metric.
3. **FOV sweep:** same prompt/seed at several FOV values; verify repeatable
   projection/framing association rather than only pixel differences. Pin the
   subject and composition in the evaluation prompts.
4. **Cross-camera holdout:** equivalent FOV from unseen camera or camera+lens
   configurations should produce similar observational control.
5. **Resolution transfer:** run counterfactuals at 512 and representative 1024
   buckets. Do not change the no-crop training policy merely to match a 512-only
   diagnostic.
6. **Text and quality preservation:** short/long prompt grids, FID/KID and
   collapse metrics must stay within the P1 tolerance.

## Go/No-Go Gates

Proceed from P3a to P3b only when:

- unknown conditioning is exactly zero after training;
- guided-prediction sensitivity is repeatable across held-out prompts;
- generated framing changes in the requested direction;
- the result is not confined to one camera model or one scene group.

P3a is an association/memorization falsifier. P3b measures observational
cross-source generalization. Claim physical camera control only after P3c paired
or synthetic validation; the local-only result is a feasibility signal, and
raw-focal Unsplash training is only a correlation experiment.

## Reproducibility And Schema

Version the camera condition schema by semantic field names, normalization,
capture-head policy and centering rule, and require exact equality when loading
a camera checkpoint. A matching tensor shape is insufficient.

Commit an audit script and versioned camera sensor table. Each audit produces a
JSON summary containing field/confidence counts, camera and lens distributions,
dimension mismatch rejections, caption leakage counts, source-manifest hashes,
sensor-table hash and audit-tool version. Private image paths or images do not
need to be committed; the rules and aggregate artifact do.

The initial strict local audit is committed as
[`artifacts/pictures_training_camera_audit_v2.json`](artifacts/pictures_training_camera_audit_v2.json).
