using System;
using UnityEngine;
using CPritch.DepthForge.Editor.Data;

namespace CPritch.DepthForge.Editor.Inference
{
    /// <summary>
    /// Which surface maps a provider emits *natively* (straight from the model) vs. what the host has
    /// to derive. This is the seam that decouples the tool from "the model produces height": Depth
    /// Anything is Height-only (normal/AO derived), the DepthForge material model is
    /// Height|Normal|Roughness (only AO derived).
    /// </summary>
    [Flags]
    public enum MapKinds
    {
        None = 0,
        Height = 1,
        Normal = 2,
        Roughness = 4,
        AO = 8
    }

    /// <summary>
    /// The native maps an <see cref="IMapProvider"/> produced for one source. Any field may be null;
    /// <see cref="native"/> records which ones are genuinely model-produced (the rest the host fills
    /// in via <c>MapPipeline</c>). Textures here are transient and owned by the caller.
    /// </summary>
    public class MapSet
    {
        public Texture2D height;
        public Texture2D normal;
        public Texture2D roughness;
        public MapKinds native;

        public bool Has(MapKinds kind) => (native & kind) != 0;
    }

    /// <summary>Identity + licence posture of a provider, used for the dropdown and packaging gates.</summary>
    public class ProviderInfo
    {
        public string id;             // stable key persisted in Recipe.providerId
        public string displayName;    // shown in the provider dropdown
        public bool commercialClean;  // false for Depth Anything (CC-BY-NC)
        public bool referenceOnly;    // true => excluded from shippable builds (DEPTHFORGE_SHIPPING)
        public string licenseNote;    // human-readable licence summary
    }

    /// <summary>
    /// A model that turns a single source texture into a <see cref="MapSet"/>. Owns its own model
    /// identity, input contract and (a)sync inference. The host never assumes which maps come back —
    /// it reads <see cref="NativeMaps"/> and derives the remainder.
    /// </summary>
    public interface IMapProvider : IDisposable
    {
        ProviderInfo Info { get; }

        /// <summary>Capability flags — which maps this provider emits natively.</summary>
        MapKinds NativeMaps { get; }

        bool IsBusy { get; }

        /// <summary>True if the model files are present locally and ready to run; otherwise
        /// <paramref name="reason"/> carries a user-facing explanation (e.g. "model still in training,
        /// not yet bundled", or "model not downloaded yet"). A cheap check — no downloads.</summary>
        bool IsAvailable(out string reason);

        /// <summary>
        /// Ensures the model is present, downloading it if the provider supports that. Drives the
        /// host's progress UI via <paramref name="onProgress"/> (value in [0,1] + a status line),
        /// then calls <paramref name="onReady"/> once runnable, or <paramref name="onError"/> with a
        /// user-facing message (including "not downloadable" cases like a model still in training).
        /// </summary>
        void PrepareAsync(Recipe recipe, Action<float, string> onProgress, Action onReady, Action<string> onError);

        /// <summary>Loads the worker for the model implied by <paramref name="recipe"/>. The provider
        /// resolves its own backend from <c>recipe.backend</c> + its own rules. Call after the model
        /// is available (<see cref="PrepareAsync"/>). Reused once per batch.</summary>
        void Initialize(Recipe recipe);

        /// <summary>Cooperative, non-blocking inference. Reports progress in [0,1] and returns the
        /// provider's native <see cref="MapSet"/> on success.</summary>
        void ExecuteAsync(Texture2D input, Recipe recipe,
            Action<float> onProgress, Action<MapSet> onDone, Action<string> onError);
    }
}
