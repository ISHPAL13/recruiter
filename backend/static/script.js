window.initializeInterviewLogic = function () {
  const BACKEND_BASE = window.location.origin;

  const mediaElement = document.getElementById("mediaElement");
  const userWebcam = document.getElementById("userWebcam");
  const taskInput = document.getElementById("taskInput");
  const startBtn = document.getElementById("startBtn");
  const stopSpeakingBtn = document.getElementById("stopSpeakingBtn");
  const closeBtn = document.getElementById("closeBtn");
  const evaluateBtn = document.getElementById("evaluateBtn");
  const conversationState = document.getElementById("conversationState");
  const statusEl = document.getElementById("status");
  const liveTranscriptBox = document.getElementById("liveTranscriptBox");
  const liveTranscript = document.getElementById("liveTranscript");
  const talkBtn = document.getElementById("talkBtn");
  const micTestBtn = document.getElementById("micTestBtn");
  const micTestResult = document.getElementById("micTestResult");
  const micTranscript = document.getElementById("micTranscript");
  const transcriptText = document.getElementById("transcriptText");

  if (!startBtn || !statusEl) return;

  let ws = null;
  let mediaStream = null;
  let playbackContext = null;
  let captureContext = null;
  let captureSource = null;
  let captureProcessor = null;
  let nextPlayTime = 0;
  let isStartingSession = false;
  let conversationActive = false;
  let currentModelTranscript = "";
  let currentUserTranscript = "";
  let activeSources = [];

  function logStatus(message) {
    const ts = new Date().toLocaleTimeString();
    statusEl.innerHTML += `[${ts}] ${message}\n`;
    statusEl.scrollTop = statusEl.scrollHeight;
    console.log(message);
  }

  function updateConversationState(message) {
    if (conversationState) {
      conversationState.innerHTML = `<span style="display: block; width: 8px; height: 8px; background-color: #10b981; border-radius: 50%;"></span>${message}`;
    }
  }

  function renderTranscript() {
    if (!liveTranscript || !liveTranscriptBox) return;
    liveTranscriptBox.style.display = "block";
    const userLine = currentUserTranscript ? `<div><strong>You:</strong> ${currentUserTranscript}</div>` : "";
    const modelLine = currentModelTranscript ? `<div style="margin-top:8px;"><strong>Sarah:</strong> ${currentModelTranscript}</div>` : "";
    liveTranscript.innerHTML = userLine || modelLine
      ? `${userLine}${modelLine}`
      : '<span class="text-gray-500 italic">Waiting for speech...</span>';
  }

  async function getUserInfo() {
    const res = await fetch(`${BACKEND_BASE}/api/user-info`);
    return res.json();
  }

  async function ensureMediaStream() {
    if (mediaStream) return mediaStream;

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });

    if (mediaElement) {
      mediaElement.srcObject = mediaStream;
      mediaElement.muted = true;
      mediaElement.play().catch(() => {});
    }

    if (userWebcam) {
      userWebcam.srcObject = mediaStream;
      userWebcam.muted = true;
      userWebcam.play().catch(() => {});
    }

    logStatus("Webcam and microphone initialized");
    return mediaStream;
  }

  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  async function startCapturePipelines(stream) {
    if (playbackContext || captureContext) {
      return;
    }

    playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    captureContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    if (playbackContext.state === "suspended") {
      await playbackContext.resume();
    }
    if (captureContext.state === "suspended") {
      await captureContext.resume();
    }

    captureSource = captureContext.createMediaStreamSource(stream);
    captureProcessor = captureContext.createScriptProcessor(2048, 1, 1);
    captureProcessor.onaudioprocess = (event) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const float32 = event.inputBuffer.getChannelData(0);
      const int16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      ws.send(JSON.stringify({ audio: arrayBufferToBase64(int16.buffer) }));
    };
    captureSource.connect(captureProcessor);
    captureProcessor.connect(captureContext.destination);
  }

  function playAudioChunk(base64Audio) {
    if (!playbackContext) return;
    if (playbackContext.state === "suspended") {
      playbackContext.resume().catch(() => {});
    }

    const binaryString = atob(base64Audio);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const int16 = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }

    const audioBuffer = playbackContext.createBuffer(1, float32.length, 24000);
    audioBuffer.getChannelData(0).set(float32);

    const source = playbackContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(playbackContext.destination);
    source.onended = () => {
      activeSources = activeSources.filter((item) => item !== source);
    };

    const now = playbackContext.currentTime;
    if (nextPlayTime < now) nextPlayTime = now;
    source.start(nextPlayTime);
    nextPlayTime += audioBuffer.duration;
    activeSources.push(source);
  }

  function clearQueuedAudio() {
    activeSources.forEach((source) => {
      try {
        source.stop();
      } catch {}
    });
    activeSources = [];
    if (playbackContext) {
      nextPlayTime = playbackContext.currentTime;
    } else {
      nextPlayTime = 0;
    }
  }

  async function connectGeminiSession() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/interview`);

    ws.onopen = () => {
      logStatus("Connected to interview stream");
      updateConversationState("Connecting to Gemini...");
    };

    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data);

      if (data.error) {
        logStatus(`Error: ${data.error}`);
        return;
      }

      if (data.status === "connected") {
        conversationActive = true;
        updateConversationState("Gemini live interview active");
        logStatus("Gemini Live session connected");
        return;
      }

      if (data.inputTranscript) {
        currentUserTranscript = data.inputTranscript;
        renderTranscript();
      }

      if (data.text) {
        currentModelTranscript += data.text;
        renderTranscript();
      }

      if (data.audio) {
        playAudioChunk(data.audio);
      }

      if (data.interrupted) {
        clearQueuedAudio();
        logStatus("Gemini response interrupted");
      }

      if (data.turnComplete) {
        if (currentUserTranscript) {
          logStatus(`You: ${currentUserTranscript}`);
        }
        if (currentModelTranscript) {
          logStatus(`Sarah: ${currentModelTranscript}`);
        }
        currentUserTranscript = "";
        currentModelTranscript = "";
        renderTranscript();
        updateConversationState("Listening...");
      }
    };

    ws.onerror = () => {
      logStatus("Interview stream connection error");
    };

    ws.onclose = () => {
      conversationActive = false;
      updateConversationState("Session closed");
      logStatus("Interview stream closed");
    };
  }

  async function startSessionFlow() {
    if (isStartingSession || conversationActive) {
      logStatus("Session already in progress");
      return;
    }

    try {
      isStartingSession = true;
      startBtn.disabled = true;
      if (evaluateBtn) evaluateBtn.style.display = "flex";

      const userInfo = await getUserInfo();
      if (!userInfo.logged_in) {
        window.location.href = "/login";
        return;
      }

      currentUserTranscript = "";
      currentModelTranscript = "";
      renderTranscript();
      updateConversationState("Preparing devices...");
      const stream = await ensureMediaStream();
      await startCapturePipelines(stream);
      await connectGeminiSession();
    } catch (error) {
      logStatus(`Start error: ${error.message || error}`);
      startBtn.disabled = false;
    } finally {
      isStartingSession = false;
    }
  }

  async function stopSessionFlow() {
    conversationActive = false;
    clearQueuedAudio();

    if (captureProcessor) {
      captureProcessor.disconnect();
      captureProcessor = null;
    }

    if (captureSource) {
      captureSource.disconnect();
      captureSource = null;
    }

    if (captureContext) {
      captureContext.close();
      captureContext = null;
    }

    if (playbackContext) {
      playbackContext.close();
      playbackContext = null;
    }

    nextPlayTime = 0;

    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }

    if (mediaElement) mediaElement.srcObject = null;
    if (userWebcam) userWebcam.srcObject = null;

    if (ws) {
      ws.close();
      ws = null;
    }

    currentUserTranscript = "";
    currentModelTranscript = "";
    renderTranscript();
    updateConversationState("Ready to Start");
    logStatus("Session stopped");
    startBtn.disabled = false;
  }

  function sendManualText() {
    const text = (taskInput?.value || "").trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ text }));
    taskInput.value = "";
    logStatus(`You typed: ${text}`);
  }

  async function evaluateAndGenerateReport() {
    if (!evaluateBtn) return;
    try {
      evaluateBtn.disabled = true;
      const response = await fetch(`${BACKEND_BASE}/api/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" }
      });
      const result = await response.json();
      if (!result.success) {
        throw new Error(result.error || "Evaluation failed");
      }

      logStatus(`Evaluation complete. Overall score: ${result.evaluation.overall_score}/100`);

      const a = document.createElement("a");
      a.href = `${BACKEND_BASE}/api/download-report/${result.report_filename}`;
      a.download = "Interview_Evaluation_Report.pdf";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (error) {
      logStatus(`Evaluation error: ${error.message || error}`);
    } finally {
      evaluateBtn.disabled = false;
    }
  }

  async function manualStopSpeaking() {
    clearQueuedAudio();
    updateConversationState("Listening...");
    logStatus("Stopped current Gemini audio playback");
  }

  async function runMicTest() {
    if (!micTestBtn || !micTestResult || !transcriptText || !micTranscript) return;
    try {
      micTestBtn.disabled = true;
      micTestResult.style.display = "block";
      micTranscript.style.display = "block";
      micTestResult.textContent = "Requesting microphone access...";
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micTestResult.textContent = "Microphone access granted";
      transcriptText.textContent = "Microphone is available for Gemini Live.";
      stream.getTracks().forEach((track) => track.stop());
    } catch (error) {
      micTestResult.textContent = `Microphone error: ${error.message || error}`;
      transcriptText.textContent = "Unable to access microphone.";
    } finally {
      micTestBtn.disabled = false;
    }
  }

  startBtn.addEventListener("click", startSessionFlow);
  stopSpeakingBtn?.addEventListener("click", manualStopSpeaking);
  closeBtn?.addEventListener("click", stopSessionFlow);
  talkBtn?.addEventListener("click", sendManualText);
  evaluateBtn?.addEventListener("click", evaluateAndGenerateReport);
  taskInput?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      sendManualText();
    }
  });
  micTestBtn?.addEventListener("click", runMicTest);

  window.__hiregenieInterviewCleanup = stopSessionFlow;
};
