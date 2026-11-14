"""Station digital twin: real D415 cameras + arm joints -> MuJoCo scene.

Pipeline: ``station`` (Intel, owns the D415s) -> ``perception`` (RGB-D -> StationState)
-> HTTP JSON -> ``sim_sync`` (StationState -> MuJoCo World). ``state`` is the shared contract.
"""
