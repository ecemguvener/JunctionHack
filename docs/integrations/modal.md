# Modal — isolated, reproducible PatchCore training experiments

`cloud/integrations/modal_patchcore.py` provides a real Modal function running
Brickcell's existing train/validation-calibration command in a separate Linux
CPU environment. It uses the same Python 3.12 lock and Anomalib 2.7.0, an 8-CPU/
16-GB worker, a one-hour deadline and an explicit persistent experiment volume.
It does not add CUDA to the Mac, reinterpret a cloud score or evaluate test data.
The existing unrelated `cloud/modal_app.py` policy service remains untouched.

Authenticate Modal in a separate environment, create/upload your own explicit
training config, manifest and images into `brickcell-inspection-experiments`,
preserving their relative paths. Keep object/session groups disjoint, train only
on good images and reserve val/test partitions. The worker runs the existing
strict manifest validator and rejects image/config/output paths outside its volume
or an existing model output. Do not upload datasets without permission.

Run from the repository root in that separate Modal environment:

```bash
modal run cloud/integrations/modal_patchcore.py --config configs/experiment.yaml
```

The first explicit run builds the locked image and may create the experiment
volume; it consumes Modal credits. It uploads source code only from the local
repository, not local artifacts or environment files. Successful output includes
job/config/manifest/image hashes, synthetic status and checkpoint hash. Training
logs and `remote-job.json` persist in the volume. Configure `torch_threads: 8`
when appropriate. Export/parity and a once-only frozen test evaluation are separate
steps, not automatically run by this worker. GPU training would require an
explicit separate environment/model change and new validation; it is not claimed.

Offline tests exercise data contracts, bounded paths and refusing model overwrite.
The Modal file is syntax-checked without importing its SDK. No cloud image build,
account authentication, dataset upload, training job or speedup was run here.
[Modal functions and volumes](https://modal.com/docs/guide).
