using NativeWebSocket;
using SimpleJSON; 
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.UI;


public class GeminiLiveClient : MonoBehaviour
{
    #region Unity
    public TMPro.TMP_InputField inputField; 
    public TextMeshProUGUI recordingBtnText;
    #endregion


    [Header("WebSocket Server Configuration")]
    public string serverUrl = "ws://127.0.0.1:8765";

    [Header("Streaming Settings")]
    public float imageSendInterval = 1.0f;
    private const int RECORDING_MAX_DURATION = 60;

    [Header("Audio Settings")]
    private const int INPUT_SAMPLE_RATE = 16000;
    private const int OUTPUT_SAMPLE_RATE = 24000;

    private WebSocket ws;
    private bool isRecording = false;
    private AudioClip inputAudioClip;
    private int audioPosition = 0;
    private Queue<byte[]> audioQueue = new Queue<byte[]>();

    private int recordedSamplePosition = 0;
    private AudioSource audioSource; 
    private Coroutine playbackCoroutine;
    private Coroutine audioStreamingCoroutine; 

    private string lastHandle= string.Empty;

    void Start()
    {
        audioSource = GetComponent<AudioSource>();
        if (audioSource == null)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
        }

        Connect();

        playbackCoroutine = StartCoroutine(ProcessAudioQueueCoroutine());
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (ws != null)
        {
            ws.DispatchMessageQueue();
        }
#endif
    }

    private void OnDestroy()
    {
        StopMicrophone();
        if (ws != null && ws.State == WebSocketState.Open)
        {
            ws.Close();
        }
        if (playbackCoroutine != null)
        {
            StopCoroutine(playbackCoroutine);
        }
    }
    
    private async void Connect()
    {
        var finalUrl= serverUrl;
        if(string.IsNullOrEmpty(lastHandle)==false)
        {
            finalUrl += $"/{lastHandle}";
        }

        ws = new WebSocket(finalUrl);

        ws.OnOpen += () =>
        {
            Debug.Log("WebSocket Successed!");
        };

        ws.OnError += (e) =>
        {
            Debug.LogError("WebSocket Error: " + e);
        };

        ws.OnClose += (e) =>
        {
            Debug.Log("WebSocket Closed: " + e);
            StopMicrophone();
            StopAllCoroutines();
        };

        ws.OnMessage += OnMessageReceived;

        await ws.Connect();
    }

    public void SendInputText()
    {
        if(inputField!=null)
        {
            Debug.Log("Sending Text: " + inputField.text);

            string textJson = JsonUtility.ToJson(new TextMessage
            {
                type = "text",
                content = inputField.text,
            });

            Send(textJson);
            inputField.text = "";
        }
    }

    public void SendScreenStreaming()
    {
        StartCoroutine(ScreenImageStreamingCoroutine());
    }

    private async void Send(string message)
    {
        if (ws != null && ws.State == WebSocketState.Open)
        {
            await ws.SendText(message);
        }
    }

    private async void Send(byte[] bytes)
    {
        if (ws != null && ws.State == WebSocketState.Open)
        {
            await ws.Send(bytes);
        }
    }


    private async void OnMessageReceived(byte[] bytes)
    {
        string message = Encoding.UTF8.GetString(bytes);

        var N = JSON.Parse(message);
        string type = N["type"];
        string content = N["content"];

        if (type == "text")
        {
            Debug.Log($"Gemini (Text): {content}");
        }
        else if (type == "audio")
        {
            try
            {
                byte[] rawAudioBytes = System.Convert.FromBase64String(content);
                AppendAudioData(rawAudioBytes);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Decode Base64 AudioData Failed: {e.Message}");
            }
        }
        else if(type== "update_handle")
        {
            lastHandle = content;
        }
        else if(type=="reconnect")
        {
            float time = 0;
            if(float.TryParse(content,out time)==false)
            {
                time = 0;
            }
            await Task.Delay(TimeSpan.FromSeconds(time));
            await ws.Close();
            Debug.Log("Server Request Reconnect");
            Connect();
        }
    }

    private IEnumerator ScreenImageStreamingCoroutine()
    {
        while (ws != null && ws.State == WebSocketState.Open)
        {
            yield return new WaitForEndOfFrame();

            Texture2D screenTexture = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
            screenTexture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
            screenTexture.Apply();

            byte[] jpegBytes = screenTexture.EncodeToJPG(quality: 75);
            string base64Image = Convert.ToBase64String(jpegBytes);

            string imageJson = JsonUtility.ToJson(new ImageMessage
            {
                type = "image",
                data = base64Image,
                mime_type = "image/jpeg"
            });

            Debug.Log("Start Sending Image");
            Send(imageJson);

            Destroy(screenTexture);

            yield return  new WaitForSeconds(1);
        }
    }

    public void ToggleRecording()
    {
        if (isRecording)
        {
            recordedSamplePosition = Microphone.GetPosition(null);

            recordingBtnText.text = "StartRecord";
            StopMicrophone();
            StartCoroutine(SendRecordedAudioCoroutine());
        }
        else
        {
            recordingBtnText.text =  "Recording...";
            StartMicrophone();
        }
    }

    private void StartMicrophone()
    {
        if (isRecording) return;

        StopMicrophone();

        inputAudioClip = Microphone.Start(null, true, RECORDING_MAX_DURATION, INPUT_SAMPLE_RATE);
        isRecording = true;
        Debug.Log("StartRecording，Sample Rate: " + INPUT_SAMPLE_RATE + "Hz");

        audioStreamingCoroutine = StartCoroutine(CheckRecordingEnd());
    }

    private void StopMicrophone()
    {
        if (!isRecording) return;

        Microphone.End(null);
        isRecording = false;

        if (audioStreamingCoroutine != null)
        {
            StopCoroutine(audioStreamingCoroutine);
            audioStreamingCoroutine = null;
        }

        Debug.Log("Stop Recordinng");
    }

    private IEnumerator CheckRecordingEnd()
    {
        yield return new WaitForSeconds(RECORDING_MAX_DURATION);

        if (isRecording)
        {
            Debug.LogWarning("Recording reached max duration. Stopping and sending automatically.");
            recordedSamplePosition = Microphone.GetPosition(null);
            StopMicrophone();
            StartCoroutine(SendRecordedAudioCoroutine());
            recordingBtnText.text = "StartRecord";
        }
    }

    /// <summary>
    /// Get PCM data from AudioClip
    /// </summary>
    private IEnumerator SendRecordedAudioCoroutine()
    {
        yield return new WaitForSeconds(1); 

        if (inputAudioClip == null) yield break; 

        int finalPosition = recordedSamplePosition;

        if (finalPosition == 0)
        {
            Debug.LogWarning("clip is to short!");
            Destroy(inputAudioClip);
            inputAudioClip = null;
            yield break; 
        }


        float[] fullSamples = new float[inputAudioClip.samples * inputAudioClip.channels];
        inputAudioClip.GetData(fullSamples, 0);

        float[] recordedSamples = new float[finalPosition * inputAudioClip.channels];
        Array.Copy(fullSamples, recordedSamples, recordedSamples.Length);

        byte[] pcmBytes = ConvertFloatToPcm16(recordedSamples);

        Debug.Log($"Send pcm bytes: {pcmBytes.Length} bytes. count: {finalPosition}");
        Send(pcmBytes); 

        Destroy(inputAudioClip);
        inputAudioClip = null;
        recordedSamplePosition = 0;
    }

    private void AppendAudioData(byte[] newBytes)
    {
        audioQueue.Enqueue(newBytes);
        Debug.Log($"Add {newBytes.Length} bytes to the queue。current queue count: {audioQueue.Count}");
    }

    private IEnumerator ProcessAudioQueueCoroutine()
    {
        while (true)
        {
            yield return new WaitWhile(() => audioQueue.Count == 0);
            byte[] rawAudioBytes = audioQueue.Dequeue();
            float[] samples = ConvertPcm16ToFloat(rawAudioBytes);

            AudioClip clip = AudioClip.Create("GeminiResponseChunk", samples.Length, 1, OUTPUT_SAMPLE_RATE, false);
            clip.SetData(samples, 0);

            audioSource.clip = clip;
            audioSource.loop = false;
            audioSource.Play();

            float clipDuration = clip.length;
            yield return new WaitForSeconds(clipDuration);

            Destroy(clip);
        }
    }

    /// <summary>
    /// convert Unity  float[] audio data -----> 16-bit PCM (byte[])
    /// </summary>
    private byte[] ConvertFloatToPcm16(float[] floatArray)
    {
        Int16[] intData = new Int16[floatArray.Length];
        Byte[] bytes = new Byte[floatArray.Length * 2]; // 16-bit = 2 bytes

        for (int i = 0; i < floatArray.Length; i++)
        {
            intData[i] = (Int16)(floatArray[i] * Int16.MaxValue);
            Byte[] byteArr = BitConverter.GetBytes(intData[i]);
            byteArr.CopyTo(bytes, i * 2);
        }
        return bytes;
    }

    /// <summary>
    /// convert  16-bit PCM (byte[]) -----> Unity  float[] audio data
    /// </summary>
    private float[] ConvertPcm16ToFloat(byte[] byteArray)
    {
        if (byteArray.Length % 2 != 0)
        {
            Debug.LogError("PCM Data Length is error。");
            return new float[0];
        }

        float[] floatArray = new float[byteArray.Length / 2];

        for (int i = 0; i < floatArray.Length; i++)
        {
            Int16 sample = BitConverter.ToInt16(byteArray, i * 2);
            floatArray[i] = (float)sample / Int16.MaxValue;
        }
        return floatArray;
    }


    public void SendTextMessage(string text)
    {
        string textJson = JsonUtility.ToJson(new TextMessage
        {
            type = "text",
            content = text
        });

        Send(textJson);
    }
}

[System.Serializable]
public class TextMessage
{
    public string type;
    public string content;
}

[System.Serializable]
public class ImageMessage
{
    public string type;
    public string data; 
    public string mime_type;
}