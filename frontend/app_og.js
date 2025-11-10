// --- config ---
const BACKEND_URL = "http://127.0.0.1:5000";

// --- state ---
let stream = null, isProcessing = false, selectedLanguage = "Hindi";
let translationCount = 0, sessionStart, latencies = [];

// --- elements ---
const video = document.getElementById("videoFeed");
const overlay = document.getElementById("overlay");
const recording = document.getElementById("recording");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const captureBtn = document.getElementById("captureBtn");
const islOutput = document.getElementById("islOutput");
const correctedOutput = document.getElementById("correctedOutput");
const translationOutput = document.getElementById("translationOutput");
const confidenceBar = document.getElementById("confidenceBar");
const speakBtn = document.getElementById("speakBtn");
const volume = document.getElementById("volume");
const statusText = document.getElementById("statusText");
const translationCountEl = document.getElementById("translationCount");
const avgLatencyEl = document.getElementById("avgLatency");
const accuracyEl = document.getElementById("accuracy");
const uptimeEl = document.getElementById("uptime");
const toast = document.getElementById("toast");

// --- helpers ---
const showToast = (msg) => { toast.textContent = msg; toast.classList.remove("hidden"); setTimeout(()=>toast.classList.add("hidden"), 2000); };
const avg = (a)=> a.length? Math.round(a.reduce((x,y)=>x+y,0)/a.length):0;
const fmtMs = (ms)=> `${ms}ms`;

// health ping
fetch(`${BACKEND_URL}/health`).then(r=>r.json()).then(d=>{
  statusText.textContent = `System Active (${d.provider})`;
}).catch(()=>{ statusText.textContent = "Backend offline"; });

// landing scroll
document.getElementById("scrollBtn").onclick = ()=> document.getElementById("app").scrollIntoView({behavior:"smooth"});

// language buttons
document.querySelectorAll(".lang").forEach(btn=>{
  btn.onclick = ()=>{
    document.querySelectorAll(".lang").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    selectedLanguage = btn.dataset.lang;
    if (correctedOutput.textContent && correctedOutput.textContent !== "-") {
      translateNow(correctedOutput.textContent);
    }
  };
});

// camera
startBtn.onclick = async ()=>{
  try{
    stream = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    video.srcObject = stream;
    overlay.style.display = "none";
    startBtn.disabled = true; stopBtn.disabled = false; captureBtn.disabled = false;
    speakBtn.disabled = false;
    sessionStart = Date.now();
    tickUptime();
    showToast("Camera started");
  }catch(e){ showToast("Camera permission denied"); }
};
stopBtn.onclick = ()=>{
  if(stream){ stream.getTracks().forEach(t=>t.stop()); }
  stream=null; overlay.style.display="grid";
  startBtn.disabled = false; stopBtn.disabled = true; captureBtn.disabled = true;
};
speakBtn.onclick = ()=>{
  const text = translationOutput.textContent === "-" ? correctedOutput.textContent : translationOutput.textContent;
  if(!text || text === "-") return;
  const u = new SpeechSynthesisUtterance(text);
  u.volume = parseInt(volume.value,10)/100;
  window.speechSynthesis.speak(u);
};

// capture → predict → autocorrect → translate
captureBtn.onclick = async ()=>{
  if(!video || isProcessing) return;
  isProcessing = true; recording.classList.add("on");
  const t0 = performance.now();

  try{
    // frame to jpeg b64
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video,0,0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);

    // predict
    const p = await fetch(`${BACKEND_URL}/predict`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({image: dataUrl})
    }).then(r=>r.json());
    if(!p.success) throw new Error(p.error||"predict failed");

    islOutput.classList.remove("placeholder");
    islOutput.textContent = p.prediction || "NO_SIGN_DETECTED";
    confidenceBar.style.width = `${Math.round((p.confidence||0)*100)}%`;
    accuracyEl.textContent = `${Math.round((p.confidence||0)*100)}%`;

    // autocorrect
    const ac = await fetch(`${BACKEND_URL}/autocorrect`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({text: islOutput.textContent})
    }).then(r=>r.json());
    if(!ac.success) throw new Error(ac.error||"autocorrect failed");
    correctedOutput.classList.remove("placeholder");
    correctedOutput.textContent = ac.corrected_text || islOutput.textContent;

    // translate
    await translateNow(correctedOutput.textContent);

    // stats
    const dt = Math.round(performance.now()-t0);
    latencies.push(dt);
    avgLatencyEl.textContent = fmtMs(avg(latencies));
    translationCount += 1; translationCountEl.textContent = translationCount;

  }catch(e){
    showToast(e.message||String(e));
  }finally{
    isProcessing = false; recording.classList.remove("on");
  }
};

async function translateNow(text){
  const tr = await fetch(`${BACKEND_URL}/translate`, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({text, language:selectedLanguage})
  }).then(r=>r.json());
  if(!tr.success) throw new Error(tr.error||"translate failed");
  translationOutput.classList.remove("placeholder");
  translationOutput.textContent = tr.translation || text;
}

function tickUptime(){
  if(!sessionStart) return;
  const sec = Math.floor((Date.now()-sessionStart)/1000);
  const m = String(Math.floor(sec/60)); const s = String(sec%60).padStart(2,"0");
  uptimeEl.textContent = `${m}:${s}`;
  requestAnimationFrame(tickUptime);
}