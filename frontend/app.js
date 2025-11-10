// frontend/app.js

async function startCamera() {
  if (stream) return stream;
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width:  { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user"
    },
    audio: false
  });
  videoFeed.srcObject = stream;
  await videoFeed.play();
  return stream;
}

(() => {
  const BACKEND_URL = "http://127.0.0.1:5000"; // change if your Flask runs elsewhere

  // -------- DOM refs (must exist in HTML) --------
  const videoFeed         = document.getElementById("videoFeed");
  const islOutput         = document.getElementById("islOutput");
  const correctedOutput   = document.getElementById("correctedOutput");
  const translationOutput = document.getElementById("translationOutput");

  // Optional controls (only wired if present)
  const finalizeBtn    = document.getElementById("finalizeBtn");
  const undoBtn        = document.getElementById("undoBtn");
//   const clearBtn       = document.getElementById("clearBtn");
  const speedBeginner  = document.getElementById("speedBeginner");
  const speedMedium    = document.getElementById("speedMedium");
  const speedFast      = document.getElementById("speedFast");

  // Optional language selector (fallback to Hindi)
  const languageSelect = document.getElementById("languageSelect");

  // -------- Speed / stability --------
  const SPEED = {
    beginner: { STABLE_MS: 1200, DEBOUNCE_MS: 360 },
    medium:   { STABLE_MS: 1000, DEBOUNCE_MS: 300 },
    fast:     { STABLE_MS: 800,  DEBOUNCE_MS: 240 },
  };
  let CURRENT_SPEED = "medium";
  let STABLE_MS   = SPEED[CURRENT_SPEED].STABLE_MS;
  let DEBOUNCE_MS = SPEED[CURRENT_SPEED].DEBOUNCE_MS;

  function markSpeed(mode){
    document.getElementById("speedBeginner")?.classList.toggle("active", mode==="beginner");
    document.getElementById("speedMedium")?.classList.toggle("active",   mode==="medium");
    document.getElementById("speedFast")?.classList.toggle("active",     mode==="fast");
  }
  function setSpeed(mode){
    if(!SPEED[mode]) return;
    CURRENT_SPEED = mode;
    STABLE_MS   = SPEED[mode].STABLE_MS;
    DEBOUNCE_MS = SPEED[mode].DEBOUNCE_MS;
    markSpeed(mode);
  }
  // default highlight
  markSpeed("medium");

  // -------- State --------
  let stream = null;
  let loopOn = false;
  let lastCommit = 0;            // ms
  let currentWord = "";          // building word
  let chrCount = {};             // {char: count}
  let currentMax = null;         // char with max count
  let stableStart = 0;           // ms

  function now(){ return performance.now(); }
  function resetStability(){
    chrCount = {};
    currentMax = null;
    stableStart = 0;
  }
  function updateIslOutput(){
    if (islOutput) islOutput.textContent = currentWord || "-";
  }
  function getSelectedLanguage(){
    if (languageSelect && languageSelect.value) return languageSelect.value;
    return "Hindi";
  }

  // -------- Camera helpers (optional: starts when video is playing) --------
  async function startCamera() {
    if (stream) return stream;
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    videoFeed.srcObject = stream;
    await videoFeed.play();
    return stream;
  }
  async function stopCamera() {
    loopOn = false;
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
  }

  // Auto-start prediction loop when the <video> starts playing (works with your existing Start button)
  videoFeed?.addEventListener("playing", () => {
    if (!loopOn) {
      loopOn = true;
      requestAnimationFrame(tick);
    }
  });

  // -------- Backend calls --------
  async function fetchPredictChar(frameDataUrl){
    const r = await fetch(`${BACKEND_URL}/predict_char`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ image: frameDataUrl })
    });
    return r.json();
  }

  async function finalizeWord(){
    if (!currentWord) return;
    const word = currentWord;
    currentWord = "";
    updateIslOutput();
    resetStability();

    const language = getSelectedLanguage();
    try{
      const res = await fetch(`${BACKEND_URL}/finalize_word`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ word, language })
      }).then(x => x.json());

      if (res && res.success) {
        if (res.corrected) {
          const base = correctedOutput.textContent.replace("-", "").trim();
          correctedOutput.textContent = (base ? base + " " : "") + res.corrected + " ";
        }
        if (res.translated) {
          const baseT = translationOutput.textContent.replace("-", "").trim();
          translationOutput.textContent = (baseT ? baseT + " " : "") + res.translated + " ";
        }
      }
    } catch (e){
      console.error("finalize_word failed:", e);
    }
  }

  // -------- Frame capture + stability loop --------
  function grabFrameDataURL() {
    if (!videoFeed || !videoFeed.videoWidth) return null;
    const canvas = document.createElement("canvas");
    canvas.width = videoFeed.videoWidth;
    canvas.height = videoFeed.videoHeight;
    const ctx = canvas.getContext("2d");
    // Mirror on UI only; backend model expects image as-is. If you must unmirror, remove scaleX.
    ctx.save();
    ctx.scale(1, 1);
    ctx.drawImage(videoFeed, 0, 0);
    ctx.restore();
    return canvas.toDataURL("image/jpeg", 0.8);
  }

  async function tick(){
    if (!loopOn) return;

    try{
      const frame = grabFrameDataURL();
      if (frame) {
        const res = await fetchPredictChar(frame);

        if (res.success && res.char) {
          const ch = res.char;
          chrCount[ch] = (chrCount[ch] || 0) + 1;

          const newMax = Object.keys(chrCount).sort((a,b)=>chrCount[b]-chrCount[a])[0];
          if (newMax !== currentMax) {
            currentMax = newMax;
            stableStart = now();
          } else if (!stableStart) {
            stableStart = now();
          }

          const held = now() - stableStart;
          const sinceLast = now() - lastCommit;

          if (currentMax && held >= STABLE_MS && sinceLast >= DEBOUNCE_MS) {
            currentWord += currentMax;
            lastCommit = now();
            resetStability();
            updateIslOutput();
          }
        } else {
          // no char → soften stability
          resetStability();
        }
      }
    } catch (e){
      console.error("predict_char failed:", e);
    } finally {
      // ~5 fps polling is enough for stability UI
      setTimeout(() => requestAnimationFrame(tick), 200);
    }
  }

  // -------- Buttons / Keys --------
  finalizeBtn && (finalizeBtn.onclick = finalizeWord);

  undoBtn && (undoBtn.onclick = () => {
    if (currentWord) {
      currentWord = currentWord.slice(0, -1);
      updateIslOutput();
    }
  });

  clearBtn && (clearBtn.onclick = () => {
    currentWord = "";
    updateIslOutput();
    if (correctedOutput)   correctedOutput.textContent   = "-";
    if (translationOutput) translationOutput.textContent = "-";
    resetStability();
  });

  speedBeginner && (speedBeginner.onclick = () => setSpeed("beginner"));
  speedMedium   && (speedMedium.onclick   = () => setSpeed("medium"));
  speedFast     && (speedFast.onclick     = () => setSpeed("fast"));

  window.addEventListener("keydown", (ev) => {
    if (ev.code === "Space") { ev.preventDefault(); finalizeWord(); }
    else if (ev.key === "Backspace" || ev.key === "u" || ev.key === "U") {
      if (currentWord) {
        currentWord = currentWord.slice(0, -1);
        updateIslOutput();
      }
    }
  });

  // -------- If you have Start/Stop buttons already, wire them (optional IDs) --------
  const startBtn = document.getElementById("startCameraBtn") || document.getElementById("startBtn");
  const stopBtn  = document.getElementById("stopCameraBtn")  || document.getElementById("stopBtn");

  startBtn && (startBtn.onclick = async () => {
    try { await startCamera(); } catch(e){ console.error(e); }
  });
  stopBtn && (stopBtn.onclick = async () => {
    try { await stopCamera(); } catch(e){ console.error(e); }
  });

  // -------- Init UI --------
  updateIslOutput();       // show "-" initially
  // If your page auto-starts camera elsewhere, our 'playing' listener will start the loop.
})();