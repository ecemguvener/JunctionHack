# CRACI — opt-in supply-chain build and vulnerability evidence

The workflow `.github/workflows/craci-brickcell.yml` runs a locked Python 3.12
CPU build on CRACI's GitHub Actions-compatible runners. CRACI observes build-time
dependency traffic and produces native supply-chain/SBOM evidence. Brickcell also
uploads the wheel/source archive, installed dependency inventory, Git/source
manifest and pip-audit JSON. Vulnerability findings fail the audit step while the
artifact upload still runs. A package list is not mislabeled as a complete SBOM;
consult CRACI's own completeness metadata for its observed build inventory.

Connect this GitHub repository to a CRACI account first, then manually dispatch
**CRACI - optional Brickcell core supply-chain build** on `ecem`. The workflow
has no push/pull-request trigger, so adding it does not queue paid runners or
change existing checks automatically. It has read-only repository permissions
and does not publish releases, deploy hardware, build Docker or request GPU/NPU.
Cloud credentials are not passed into it. Dependency checks may access public
package/advisory services; configure runner egress as needed.

Compatibility fix: optional MuJoCo types are now imported when simulation is
actually requested, rather than while importing the inspection package. The
workflow runs `uv run pytest` without excluding the architecture module; its
existing optional tests self-skip when MuJoCo/Frax are absent. MuJoCo/Frax stay in
their separate environment and no dependencies were added to the Mac core.
Model and live-simulator tests still require explicit assets/services.

The YAML was parsed locally and the equivalent local core checks are recorded in
`docs/VALIDATION.md`. A CRACI run, native SBOM, live advisory scan and hosted Linux
build have not been executed here. SBOM completeness and vulnerability absence
must not be inferred from the existence of this workflow.
[CRACI runner guide](https://docs.craci.com/guides/runner-workflows/).
