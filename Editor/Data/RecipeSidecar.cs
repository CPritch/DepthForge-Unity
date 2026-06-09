using System;
using System.IO;
using UnityEngine;
using UnityEditor;

namespace CPritch.DepthForge.Editor.Data
{
    /// <summary>
    /// Reads/writes a <see cref="Recipe"/> as a JSON sidecar beside its source texture
    /// (&lt;sourceName&gt;.depthforge.json). This is what makes depth work re-editable: focusing a
    /// source reloads its recipe; exporting persists it.
    /// </summary>
    public static class RecipeSidecar
    {
        public const string Extension = ".depthforge.json";

        public static string GetSidecarPath(Texture2D source)
        {
            if (source == null) return null;
            string sourcePath = AssetDatabase.GetAssetPath(source);
            if (string.IsNullOrEmpty(sourcePath)) return null;

            string dir = Path.GetDirectoryName(sourcePath);
            string name = Path.GetFileNameWithoutExtension(sourcePath);
            return Path.Combine(dir, name + Extension).Replace("\\", "/");
        }

        /// <summary>Returns the persisted recipe for a source, or null if none exists / on error.</summary>
        public static Recipe Load(Texture2D source)
        {
            string path = GetSidecarPath(source);
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) return null;

            try
            {
                string json = File.ReadAllText(path);
                return JsonUtility.FromJson<Recipe>(json);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[DepthForge] Failed to read recipe sidecar at {path}: {ex.Message}");
                return null;
            }
        }

        public static void Save(Texture2D source, Recipe recipe)
        {
            string path = GetSidecarPath(source);
            if (string.IsNullOrEmpty(path) || recipe == null) return;

            try
            {
                File.WriteAllText(path, JsonUtility.ToJson(recipe, true));
                AssetDatabase.ImportAsset(path);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[DepthForge] Failed to write recipe sidecar at {path}: {ex.Message}");
            }
        }
    }
}
