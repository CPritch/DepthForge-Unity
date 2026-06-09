using System;
using System.IO;
using UnityEngine;
using UnityEditor;

namespace CPritch.DepthForge.Editor.Data
{
    /// <summary>
    /// Local cache of the RAW (pre-adjustment) inference heightmap, keyed by source GUID, so a
    /// revisited source can restore its result and stay fully editable WITHOUT re-running inference.
    /// This is a regenerable, gitignored cache — distinct from the durable <see cref="Recipe"/>
    /// sidecar. The raw is already 8-bit grayscale, so PNG is lossless here.
    /// </summary>
    public static class RawCache
    {
        private const string CacheDir = "Assets/DepthForge/Cache";

        private static string GetPath(Texture2D source)
        {
            if (source == null) return null;
            string assetPath = AssetDatabase.GetAssetPath(source);
            if (string.IsNullOrEmpty(assetPath)) return null;
            string guid = AssetDatabase.AssetPathToGUID(assetPath);
            if (string.IsNullOrEmpty(guid)) return null;
            return $"{CacheDir}/{guid}.raw.png";
        }

        public static void SaveRaw(Texture2D source, Texture2D raw)
        {
            if (raw == null) return;
            string path = GetPath(source);
            if (string.IsNullOrEmpty(path)) return;

            try
            {
                EnsureCacheDir();
                File.WriteAllBytes(path, ImageConversion.EncodeToPNG(raw));
                // Read back via File IO (not the AssetDatabase), so no import is required.
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[DepthForge] Failed to cache raw heightmap: {ex.Message}");
            }
        }

        /// <summary>Returns the cached raw heightmap as a fresh owned Texture2D, or null.</summary>
        public static Texture2D LoadRaw(Texture2D source)
        {
            string path = GetPath(source);
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) return null;

            try
            {
                byte[] bytes = File.ReadAllBytes(path);
                var tex = new Texture2D(2, 2, TextureFormat.RGB24, false, true); // linear
                if (!tex.LoadImage(bytes)) // resizes to the PNG's dimensions
                {
                    UnityEngine.Object.DestroyImmediate(tex);
                    return null;
                }
                tex.name = "DepthForgeRawCache";
                return tex;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[DepthForge] Failed to load cached raw heightmap: {ex.Message}");
                return null;
            }
        }

        private static void EnsureCacheDir()
        {
            if (!Directory.Exists(CacheDir)) Directory.CreateDirectory(CacheDir);
            string gitignore = Path.Combine(CacheDir, ".gitignore");
            if (!File.Exists(gitignore)) File.WriteAllText(gitignore, "*\n!.gitignore\n");
        }
    }
}
