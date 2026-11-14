"""Run the existing armd API with addressed, checksummed servo reads.

The original armd source/config are unchanged. The switch uses its existing
single-read fallback during this acquisition process only. No EEPROM writes.
Run in the separate armd environment, not the MuJoCo environment.
"""

from armd.bus import Bus

original_open = Bus.open


def addressed_open(self):
    original_open(self)
    self._sync_read_enabled = False


Bus.open = addressed_open

if __name__ == "__main__":
    from armd.__main__ import main

    main()
