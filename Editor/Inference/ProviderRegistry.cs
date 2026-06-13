using System.Collections.Generic;
using CPritch.DepthForge.Editor.Inference.Providers;

namespace CPritch.DepthForge.Editor.Inference
{
    /// <summary>
    /// Creates the set of map providers and picks a sensible default. Reference-only providers
    /// (Depth Anything) are compiled out of shippable builds via <c>DEPTHFORGE_SHIPPING</c>, so a
    /// packaged build only ever sees the commercial-clean DepthForge provider.
    /// </summary>
    public static class ProviderRegistry
    {
        /// <summary>Fresh provider instances. The caller (the window) owns + disposes them.</summary>
        public static List<IMapProvider> CreateAll()
        {
            var list = new List<IMapProvider>
            {
                new DepthForgeProvider(),
            };
#if !DEPTHFORGE_SHIPPING
            list.Add(new DepthAnythingProvider());
#endif
            return list;
        }

        /// <summary>The provider to focus by default: the first *available* commercial-clean,
        /// non-reference provider; else the first available provider; else the first listed.</summary>
        public static IMapProvider Default(IList<IMapProvider> providers)
        {
            if (providers == null || providers.Count == 0) return null;

            foreach (var p in providers)
                if (p.Info.commercialClean && !p.Info.referenceOnly && p.IsAvailable(out _)) return p;
            foreach (var p in providers)
                if (p.IsAvailable(out _)) return p;
            return providers[0];
        }

        public static IMapProvider ById(IList<IMapProvider> providers, string id)
        {
            if (providers == null || string.IsNullOrEmpty(id)) return null;
            foreach (var p in providers)
                if (p.Info.id == id) return p;
            return null;
        }
    }
}
