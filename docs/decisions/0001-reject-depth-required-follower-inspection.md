# Decision: reject PR #13's depth-required inspection change

Status: rejected; pull request closed without merging.

Pull request: [#13 — Reject unsupported inspection views before scoring or completing the Plate scan](https://github.com/ecemguvener/JunctionHack/pull/13).

## Decision and reason

The repository owner rejected this proposal for the selected follower-arm
implementation. The stated direction is to proceed without depth perception
for that implementation because of concerns about simulation-to-real transfer.
This records the owner's engineering decision, not a measured comparison of
RGB and depth accuracy.

## Scope

PR #13 proposed requiring depth-supported valid inspection views before anomaly
scoring and Plate-scan completion. Its commits `a671cd6` and `a704dce` remain
unmerged as part of that rejected pull request.

This decision applies to that proposed follower/Plate inspection integration.
It does not establish that existing RGB-only acceptance shortcuts are valid;
missing evidence must not be interpreted as a successful inspection.

The rejection record itself changes no controller, detector, camera settings or
deployed Intel service. A future replacement should be proposed and reviewed
separately before deployment.
