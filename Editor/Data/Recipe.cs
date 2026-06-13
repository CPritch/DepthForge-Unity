using System;
using CPritch.DepthForge.Editor.Utils;

namespace CPritch.DepthForge.Editor.Data
{
    // Canonical model-layer enums. The transitional DepthForgeWindow keeps its own mirrored enums
    // and converts at the boundary; when the window is rebuilt (R7) it will use these directly.
    public enum DepthModelSize { Small, Base, Large } // Large slated for removal in R11 (non-commercial licence)
    public enum InferenceBackendChoice { Auto, GPUCompute, CPUBurst }
    public enum TilingAlignment { OffsetOnly, LinearRegressionFull, LinearRegressionDownscaled }

    /// <summary>
    /// Non-destructive, serializable settings that define how a Source is turned into maps.
    /// Persisted as a sidecar beside the source (see <see cref="RecipeSidecar"/>) so depth work is
    /// re-editable. NOTE: the preview-only "displacement strength" is intentionally NOT here — it is
    /// a view setting, not an output parameter (resolves the gap-analysis R1 flag).
    /// </summary>
    [Serializable]
    public class Recipe
    {
        public int version = 1;

        // Which model produced (or should produce) this recipe's maps. Empty => registry default.
        // Persisted so a source remembers its provenance across providers (see ProviderRegistry).
        public string providerId = "";

        // Generation
        public DepthModelSize modelSize = DepthModelSize.Small;
        public InferenceBackendChoice backend = InferenceBackendChoice.Auto;
        public bool tiledInference = false;
        public TilingAlignment tilingAlignment = TilingAlignment.LinearRegressionDownscaled;
        public bool letterbox = true;   // R9 — single-pass: pad to square (even scaling) instead of squashing

        // Adjustments (affect the exported height)
        public bool invert = true;
        public float contrast = 1f;
        public float midpoint = 0.5f;
        public float flatten = 0f;

        // Maps
        public bool exportNormal = true;
        public float normalStrength = 8f;
        public bool exportAO = true;    // R4 — part of the complete H+N+AO launch map set
        public float aoStrength = 1f;
        public float aoRadius = 0.02f;
        public bool exportRoughness = true; // only emitted when the active provider produces roughness natively

        // Export
        public TextureExporter.ExportFormat format = TextureExporter.ExportFormat.PNG_8Bit;
        public bool autoAssignMicroSplat = true;

        /// <summary>Deep copy — used by presets and per-job recipe edits.</summary>
        public Recipe Clone()
        {
            return (Recipe)MemberwiseClone();
        }
    }
}
