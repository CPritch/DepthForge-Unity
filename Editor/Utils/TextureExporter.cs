using UnityEngine;
using UnityEditor;
using System.IO;

namespace CPritch.DepthForge.Editor.Utils
{
    public static class TextureExporter
    {
        public enum ExportFormat
        {
            [InspectorName("PNG (8-bit)")]
            PNG_8Bit
            // EXR (true 16-bit) is parked: the map pipeline is 8-bit Color32 end to end, so an EXR
            // export today would only be an 8-bit-quantised container — misleading. Re-add once
            // height is carried as float through postprocess/adjustments (see docs/roadmap.md §5).
        }

        public static string SaveHeightmap(Texture2D generatedTexture, Texture2D sourceTexture, ExportFormat format)
        {
            if (generatedTexture == null || sourceTexture == null)
            {
                Debug.LogError("Cannot save heightmap: texture is null.");
                return null;
            }

            string sourcePath = AssetDatabase.GetAssetPath(sourceTexture);
            if (string.IsNullOrEmpty(sourcePath))
            {
                Debug.LogError("Source texture is not saved in the project.");
                return null;
            }

            string directory = Path.GetDirectoryName(sourcePath);
            string filename = Path.GetFileNameWithoutExtension(sourcePath) + "_Height";

            // PNG only for now (see ExportFormat). The `format` param is retained so the call sites
            // stay stable for when higher-precision formats return.
            string savePath = Path.Combine(directory, filename + ".png").Replace("\\", "/");
            byte[] bytes = ImageConversion.EncodeToPNG(generatedTexture);

            File.WriteAllBytes(savePath, bytes);
            AssetDatabase.Refresh();

            // Configure importer settings
            TextureImporter importer = AssetImporter.GetAtPath(savePath) as TextureImporter;
            if (importer != null)
            {
                importer.sRGBTexture = false; // Heightmaps should always be linear
                importer.mipmapEnabled = true;
                importer.textureCompression = TextureImporterCompression.Uncompressed; // Or high quality
                importer.SaveAndReimport();
            }

            Debug.Log($"Heightmap saved to: {savePath}");
            return savePath;
        }

        /// <summary>
        /// Saves a generated normal map next to the source texture as "&lt;name&gt;_Normal.png" and
        /// configures its importer as a proper Unity NormalMap (linear, mip-mapped). This gives
        /// MicroSplat real normal data instead of having it synthesize normals from the diffuse.
        /// </summary>
        public static string SaveNormalMap(Texture2D normalTexture, Texture2D sourceTexture)
        {
            if (normalTexture == null || sourceTexture == null)
            {
                Debug.LogError("Cannot save normal map: texture is null.");
                return null;
            }

            string sourcePath = AssetDatabase.GetAssetPath(sourceTexture);
            if (string.IsNullOrEmpty(sourcePath))
            {
                Debug.LogError("Source texture is not saved in the project.");
                return null;
            }

            string directory = Path.GetDirectoryName(sourcePath);
            string filename = Path.GetFileNameWithoutExtension(sourcePath) + "_Normal";
            string savePath = Path.Combine(directory, filename + ".png").Replace("\\", "/");

            byte[] bytes = ImageConversion.EncodeToPNG(normalTexture);
            File.WriteAllBytes(savePath, bytes);
            AssetDatabase.Refresh();

            TextureImporter importer = AssetImporter.GetAtPath(savePath) as TextureImporter;
            if (importer != null)
            {
                importer.textureType = TextureImporterType.NormalMap; // forces linear + correct sampling
                importer.mipmapEnabled = true;
                importer.SaveAndReimport();
            }

            Debug.Log($"Normal map saved to: {savePath}");
            return savePath;
        }

        /// <summary>
        /// Saves a generated ambient-occlusion map next to the source as "&lt;name&gt;_AO.png" and
        /// configures its importer as linear, mip-mapped grayscale.
        /// </summary>
        public static string SaveAOMap(Texture2D aoTexture, Texture2D sourceTexture)
        {
            if (aoTexture == null || sourceTexture == null)
            {
                Debug.LogError("Cannot save AO map: texture is null.");
                return null;
            }

            string sourcePath = AssetDatabase.GetAssetPath(sourceTexture);
            if (string.IsNullOrEmpty(sourcePath))
            {
                Debug.LogError("Source texture is not saved in the project.");
                return null;
            }

            string directory = Path.GetDirectoryName(sourcePath);
            string filename = Path.GetFileNameWithoutExtension(sourcePath) + "_AO";
            string savePath = Path.Combine(directory, filename + ".png").Replace("\\", "/");

            byte[] bytes = ImageConversion.EncodeToPNG(aoTexture);
            File.WriteAllBytes(savePath, bytes);
            AssetDatabase.Refresh();

            TextureImporter importer = AssetImporter.GetAtPath(savePath) as TextureImporter;
            if (importer != null)
            {
                importer.sRGBTexture = false; // AO is linear data
                importer.mipmapEnabled = true;
                importer.SaveAndReimport();
            }

            Debug.Log($"AO map saved to: {savePath}");
            return savePath;
        }

        /// <summary>
        /// Saves a generated roughness map next to the source as "&lt;name&gt;_Roughness.png" and
        /// configures its importer as linear, mip-mapped grayscale. Emitted only when the active
        /// provider produces roughness natively (e.g. the DepthForge material model).
        /// </summary>
        public static string SaveRoughness(Texture2D roughnessTexture, Texture2D sourceTexture)
        {
            if (roughnessTexture == null || sourceTexture == null)
            {
                Debug.LogError("Cannot save roughness map: texture is null.");
                return null;
            }

            string sourcePath = AssetDatabase.GetAssetPath(sourceTexture);
            if (string.IsNullOrEmpty(sourcePath))
            {
                Debug.LogError("Source texture is not saved in the project.");
                return null;
            }

            string directory = Path.GetDirectoryName(sourcePath);
            string filename = Path.GetFileNameWithoutExtension(sourcePath) + "_Roughness";
            string savePath = Path.Combine(directory, filename + ".png").Replace("\\", "/");

            byte[] bytes = ImageConversion.EncodeToPNG(roughnessTexture);
            File.WriteAllBytes(savePath, bytes);
            AssetDatabase.Refresh();

            TextureImporter importer = AssetImporter.GetAtPath(savePath) as TextureImporter;
            if (importer != null)
            {
                importer.sRGBTexture = false; // roughness is linear data
                importer.mipmapEnabled = true;
                importer.SaveAndReimport();
            }

            Debug.Log($"Roughness map saved to: {savePath}");
            return savePath;
        }
    }
}
