using UnityEngine;
using UnityEngine.Networking;
using UnityEditor;
using System;
using System.IO;

namespace CPritch.DepthForge.Editor.Utils
{
    public class DownloadHandle
    {
        private UnityWebRequest _request;
        public bool IsDone { get; private set; }
        public bool IsCancelled { get; private set; }
        public string Error { get; private set; }

        public DownloadHandle(UnityWebRequest request)
        {
            _request = request;
        }

        public void Cancel()
        {
            if (IsDone || IsCancelled) return;
            IsCancelled = true;
            _request?.Abort();
        }

        public void SetDone(string error = null)
        {
            IsDone = true;
            Error = error;
        }
    }

    public static class ModelDownloader
    {
        public static DownloadHandle DownloadModel(string url, string savePath, Action<float> onProgress, Action<string> onComplete)
        {
            if (string.IsNullOrEmpty(url) || string.IsNullOrEmpty(savePath))
            {
                onComplete?.Invoke("URL or save path is empty.");
                return null;
            }

            // Ensure directory exists
            try
            {
                string dir = Path.GetDirectoryName(savePath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }
            }
            catch (Exception ex)
            {
                onComplete?.Invoke($"Failed to create cache directory: {ex.Message}");
                return null;
            }

            string tempPath = savePath + ".tmp";
            
            // Delete existing temp file if any
            if (File.Exists(tempPath))
            {
                try { File.Delete(tempPath); } catch { }
            }

            var request = UnityWebRequest.Get(url);
            var fileHandler = new DownloadHandlerFile(tempPath);
            request.downloadHandler = fileHandler;

            var operation = request.SendWebRequest();
            var handle = new DownloadHandle(request);

            EditorApplication.CallbackFunction updateAction = null;
            updateAction = () =>
            {
                if (handle.IsCancelled)
                {
                    EditorApplication.update -= updateAction;
                    request.Dispose();
                    try { File.Delete(tempPath); } catch { }
                    onComplete?.Invoke("Download cancelled.");
                    return;
                }

                if (operation.isDone)
                {
                    EditorApplication.update -= updateAction;
                    
                    string error = null;
                    if (request.result != UnityWebRequest.Result.Success)
                    {
                        error = request.error;
                        if (string.IsNullOrEmpty(error)) error = "Unknown network error.";
                        try { File.Delete(tempPath); } catch { }
                    }
                    else
                    {
                        try
                        {
                            if (File.Exists(savePath))
                            {
                                File.Delete(savePath);
                            }
                            File.Move(tempPath, savePath);
                        }
                        catch (Exception ex)
                        {
                            error = $"Failed to finalize model file: {ex.Message}";
                        }
                    }

                    request.Dispose();
                    handle.SetDone(error);
                    onComplete?.Invoke(error);
                }
                else
                {
                    onProgress?.Invoke(operation.progress);
                }
            };

            EditorApplication.update += updateAction;
            return handle;
        }
    }
}
