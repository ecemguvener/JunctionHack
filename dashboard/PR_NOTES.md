The deployed publisher repeatedly sends cached inspection results. Previously
each POST refreshed the dashboard's five-second expiry, making old boxes appear
current. Plate-camera anomaly scores were also shown over Top-camera object
projections without distinguishing those boxes from localized defect pixels.

This follow-up to PR #10 adds capture provenance and independent acquisition
expiry. Legacy publishers remain HTTP-compatible but show “Unverified result”
with hidden boxes. Upgraded publishers retain frame/camera/time, scene mode and
algorithm metadata; heartbeats cannot extend capture freshness. Projected boxes
are labelled object estimates, and real/sim overlays are separated.

Review against `ui/twin-dashboard`, not main while PR #10 remains unmerged. The
diff intentionally avoids camera acquisition, detector algorithms, arm control,
calibration and deployment scripts. `dashboard/README.md` describes the payload
upgrade. Coordinate with the physical-controller owner before deploying because
the current publisher lacks the new evidence fields and its boxes will be hidden.

Validation: 28 Python unit/HTTP tests with fake upstreams passed; 6 Node tests of
the same overlay rules used by the page passed; JS syntax checks and Ruff passed. No physical
robot operation, detector accuracy claim or Intel deployment is part of this change.
Browser visual review has not been performed for this patch.
