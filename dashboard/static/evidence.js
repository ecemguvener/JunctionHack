/* Shared overlay rules. No result is promoted from a receive-time heartbeat. */
(function (root) {
  const api = {
    drawable(overlay, mode, statusFresh) {
      return Boolean(statusFresh && overlay?.connected && overlay?.drawable
        && overlay?.evidence?.mode === mode);
    },
    label(overlay, mode) {
      if (!overlay?.connected) return "Publisher offline";
      if (!overlay.evidence) return "Unverified result";
      if (overlay.evidence.mode !== mode) return "Different scene";
      if (!overlay.drawable) return "Evidence unavailable";
      return "Projected object estimate";
    },
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.OverlayEvidence = api;
})(globalThis);
