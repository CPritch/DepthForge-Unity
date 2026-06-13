using UnityEngine;
using CPritch.DepthForge.Editor.Data;

namespace CPritch.DepthForge.Editor.Inference
{
    /// <summary>
    /// The single place that turns a provider's native <see cref="MapSet"/> + a <see cref="Recipe"/>
    /// into the final maps, so the window preview, single export, and batch export all agree on what's
    /// native vs. derived:
    ///   • height  — always native (integration is baked into the export); host applies tuning.
    ///   • normal  — native if the provider produced one, else derived from the adjusted height.
    ///   • roughness — native only (no derivation; absent for depth-style providers).
    ///   • AO      — always derived from the adjusted height (caller does this directly).
    /// </summary>
    public static class MapPipeline
    {
        /// <summary>Adjusted height from the native height + recipe tuning. Caller owns the result.</summary>
        public static Texture2D BuildAdjustedHeight(MapSet maps, Recipe r)
        {
            if (maps == null || maps.height == null) return null;
            return ImageProcessor.ApplyAdjustments(maps.height, r.contrast, r.midpoint, r.invert, r.flatten);
        }

        /// <summary>
        /// The normal map to use. Returns the provider's native normal when present (<paramref
        /// name="derived"/> = false — do NOT destroy it, it belongs to the MapSet), otherwise a freshly
        /// derived normal from <paramref name="adjustedHeight"/> (<paramref name="derived"/> = true —
        /// caller must destroy it after use).
        /// </summary>
        public static Texture2D BuildNormal(MapSet maps, Recipe r, Texture2D adjustedHeight, out bool derived)
        {
            if (maps != null && maps.Has(MapKinds.Normal) && maps.normal != null)
            {
                derived = false;
                return maps.normal;
            }
            derived = true;
            return adjustedHeight != null ? ImageProcessor.GenerateNormalMap(adjustedHeight, r.normalStrength) : null;
        }

        /// <summary>The native roughness map, or null if this provider doesn't produce one.</summary>
        public static Texture2D NativeRoughness(MapSet maps)
        {
            return (maps != null && maps.Has(MapKinds.Roughness)) ? maps.roughness : null;
        }
    }
}
