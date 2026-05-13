import React, { useState, useEffect } from "react";

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function Icon({ name, size = 24 }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2.5,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": true,
  };

  const icons = {
    search: (
      <svg {...common}>
        <circle cx="11" cy="11" r="7" />
        <path d="M20 20l-4-4" />
      </svg>
    ),
    brain: (
      <svg {...common}>
        <path d="M9 3a3 3 0 0 0-3 3v1a3 3 0 0 0-2 5 3 3 0 0 0 2 5v1a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z" />
        <path d="M15 3a3 3 0 0 1 3 3v1a3 3 0 0 1 2 5 3 3 0 0 1-2 5v1a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3Z" />
        <path d="M7 9h3" />
        <path d="M14 9h3" />
        <path d="M8 15h2" />
        <path d="M14 15h2" />
      </svg>
    ),
    lightbulb: (
      <svg {...common}>
        <path d="M9 18h6" />
        <path d="M10 22h4" />
        <path d="M12 2a7 7 0 0 0-4 12c.8.7 1 1.5 1 2h6c0-.5.2-1.3 1-2a7 7 0 0 0-4-12Z" />
      </svg>
    ),
    chart: (
      <svg {...common}>
        <path d="M4 20V10" />
        <path d="M10 20V4" />
        <path d="M16 20v-7" />
        <path d="M22 20H2" />
      </svg>
    ),
    upload: (
      <svg {...common}>
        <path d="M12 3v13" />
        <path d="M7 8l5-5 5 5" />
        <path d="M5 21h14" />
      </svg>
    ),
    check: (
      <svg {...common}>
        <circle cx="12" cy="12" r="9" />
        <path d="M8 12l3 3 5-6" />
      </svg>
    ),
    x: (
      <svg {...common}>
        <circle cx="12" cy="12" r="9" />
        <path d="M9 9l6 6" />
        <path d="M15 9l-6 6" />
      </svg>
    ),
  };

  return icons[name] || null;
}

export default function BrainBlingTemplate() {
  const [screen, setScreen] = useState("landing");
  const [article, setArticle] = useState("");
  const [selected, setSelected] = useState("");
  const [checked, setChecked] = useState(false);
  const [message, setMessage] = useState("Paste an article or load the sample to begin.");
  const [showHintPopup, setShowHintPopup] = useState(false);
  const [metrics, setMetrics] = useState({ binary_metrics: [], ensemble_metrics: [], neural_metrics: [], nlg_metrics: null, confusion_matrices: null });
  const [question, setQuestion] = useState("");
  const [options, setOptions] = useState([]);
  const [correctAnswer, setCorrectAnswer] = useState("");
  const [inferenceTimes, setInferenceTimes] = useState({
    verification: 0,
    distractor: 0,
    hint: 0,
    total: 0,
  });
  const [modelType, setModelType] = useState("");
  const [verifierResult, setVerifierResult] = useState(null);
  const [selectedCMModel, setSelectedCMModel] = useState("Logistic Regression");
  const [theme, setTheme] = useState("multi"); // "multi" or "simple"

  // Theme color mappings
  const colors = {
    multi: {
      primary: "#bd83dd",      // purple
      secondary: "#ffc736",    // yellow
      accent: "#ff4d00",       // orange
      accent2: "#ff8bd8",      // pink
      background: "#7cf5d2",   // teal
      header: "#ff4d00",       // orange
      text: "#000",            // black
      white: "#fff",
    },
    simple: {
      primary: "#7cf5d2",      // teal
      secondary: "#ffc736",    // yellow
      accent: "#7cf5d2",       // teal
      accent2: "#ffc736",      // yellow
      background: "#ffc736",    // yellow
      header: "#7cf5d2",       // teal
      text: "#000",            // black
      white: "#fff",
    },
  };
  const c = colors[theme];

  useEffect(() => {
    fetch(`${API_BASE_URL}/metrics`)
      .then((r) => r.json())
      .then((data) => setMetrics({ binary_metrics: data.binary_metrics || [], ensemble_metrics: data.ensemble_metrics || [], neural_metrics: data.neural_metrics || [], nlg_metrics: data.nlg_metrics || null, confusion_matrices: data.confusion_matrices || null }))
      .catch((error) => {
        console.error("Failed to load metrics:", error);
        setMessage("Failed to load model metrics. Please ensure the backend is running.");
      });
  }, []);

  async function handleLoadSample() {
    setMessage("Loading random sample from dataset...");
    try {
      const res = await fetch(`${API_BASE_URL}/api/sample-article?${Date.now()}`, {
        headers: { 
          "Cache-Control": "no-cache",
          "Pragma": "no-cache"
        }
      });
      const data = await res.json();
      
      // RESET ALL QUIZ STATE to prevent caching issues
      setArticle(data.article);
      setQuestion(data.question || "");
      setOptions(data.options || []);
      setCorrectAnswer(data.correct_answer || "");
      setSelected("");
      setChecked(false);
      setMessage("Random RACE sample loaded. Click 'Submit Article' to generate quiz with distractors.");
      
      // Clear any cached hints
      window.currentHints = null;
      
      // Reset inference times
      setInferenceTimes({
        verification: 0,
        distractor: 0,
        hint: 0,
        total: 0,
      });
      
      console.log("ALL QUIZ STATE RESET - New sample loaded:", data.article.substring(0, 100));
    } catch {
      setMessage("API unavailable. Please start the backend server.");
    }
  }

  async function handleSubmitArticle() {
    if (!article.trim()) {
      setMessage("Please paste an article first.");
      return;
    }
    setSelected("");
    setChecked(false);
    setMessage("Generating quiz with distractors...");
    
    const start = performance.now();
    console.log("Starting article submission at:", start);
    try {
      // Check if this is a RACE sample by calling the sample API to get current sample
      console.log("Getting current RACE sample...");
      const sampleRes = await fetch(`${API_BASE_URL}/api/sample-article?${Date.now()}`, {
        headers: { 
          "Cache-Control": "no-cache",
          "Pragma": "no-cache"
        }
      });
      const sampleData = await sampleRes.json();
      
      // Check if current article matches the sample (simple check)
      const isRaceSample = article.trim() === sampleData.article.trim();
      
      let quizData;
      if (isRaceSample) {
        console.log("RACE sample detected, calling generate-race-quiz API...");
        // Use RACE quiz workflow: keep original correct answer, replace 3 wrong with distractors
        const raceQuizRes = await fetch(`${API_BASE_URL}/api/generate-race-quiz?${Date.now()}`, {
          method: "POST",
          headers: { 
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
          },
          body: JSON.stringify({ 
            article: sampleData.article,
            question: sampleData.question,
            options: sampleData.options.map(o => o.text),
            correct_answer: sampleData.correct_answer,
            model_type: modelType || "traditional"
          })
        });
        quizData = await raceQuizRes.json();
        console.log("RACE quiz API response received - FRESH CALL");
      } else {
        console.log("Manual article detected, calling generate-quiz API...");
        // Use manual article workflow: generate question + 1 correct + 3 distractors
        const quizRes = await fetch(`${API_BASE_URL}/api/generate-quiz`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ article, model_type: modelType || "traditional" })
        });
        quizData = await quizRes.json();
        console.log("Quiz API response received");
      }
      
      console.log("Quiz data received:", quizData);
      console.log("Correct answer from API:", quizData.correct_answer);
      setQuestion(quizData.question);
      setOptions(quizData.options.map((opt, i) => ({ id: String.fromCharCode(65 + i), text: opt })));
      setCorrectAnswer(quizData.correct_answer);
      console.log("Set correctAnswer to:", quizData.correct_answer);
      setScreen("quiz");
      setMessage(isRaceSample ? 
        "RACE quiz generated: kept original correct answer, replaced 3 wrong with distractors." :
        "Quiz generated with 1 correct option and 3 distractors."
      );
    } catch (error) {
      console.error("Error generating quiz:", error);
      setScreen("quiz");
      setMessage("API unavailable. Using fallback content.");
    }
    const end = performance.now();
    const distractorTime = Math.round(end - start);
    console.log("Quiz generation time measured:", distractorTime, "ms");
    setInferenceTimes(prev => ({
      ...prev,
      distractor: distractorTime,
      total: prev.verification + distractorTime + prev.hint,
    }));
  }

  async function handleCheckAnswer() {
    if (!selected) {
      setMessage("Please select one option before checking.");
      return;
    }
    
    // Ensure correctAnswer is set
    if (!correctAnswer) {
      console.error("correctAnswer is undefined, cannot proceed");
      setMessage("Error: Correct answer not set. Please generate a quiz first.");
      return;
    }
    
    console.log("Checking answer - selected:", selected, "correct:", correctAnswer);
    
    const start = performance.now();
    console.log("Starting answer verification at:", start);
    try {
      console.log("Calling check-answer API with model_type:", modelType);
      const res = await fetch(`${API_BASE_URL}/api/check-answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          article,
          question,
          selected_option: selected,
          options: options.map(o => o.text),
          correct_answer: correctAnswer,
          model_type: modelType || "bert"
        })
      });
      const data = await res.json();
      console.log("Check-answer API response:", data);
      setChecked(true);
      setVerifierResult(data);
      setMessage(data.explanation || "");
      setCorrectAnswer(data.correct_answer);
    } catch (error) {
      console.error("Error checking answer:", error);
      setChecked(true);
      setMessage("API unavailable. Please ensure the backend is running.");
    }
    const end = performance.now();
    const verificationTime = Math.round(end - start);
    console.log("Verification time measured:", verificationTime, "ms");
    setInferenceTimes(prev => ({
      ...prev,
      verification: verificationTime,
      total: verificationTime + prev.distractor + prev.hint,
    }));
  }

  const steps = ["landing", "input", "quiz", "dashboard"];
  const stepLabels = ["Home", "Input", "Quiz", "Results"];

  function Navbar() {
    return (
      <nav className="flex flex-wrap items-center justify-between gap-4 border-b-2 border-black px-6 py-4" style={{ backgroundColor: c.primary }}>
        <button
          onClick={() => setScreen("landing")}
          className="border-2 border-black px-5 py-2 text-3xl font-black tracking-tight shadow-[4px_4px_0px_#000]"
          style={{ backgroundColor: c.secondary }}
        >
          Brain Bling
        </button>

        <div className="flex gap-2">
          {steps.map((s, i) => (
            <button
              key={s}
              onClick={() => setScreen(s)}
              className="border-2 border-black px-4 py-2 text-sm font-bold shadow-[3px_3px_0px_#000] transition"
              style={{ backgroundColor: screen === s ? c.header : "white", color: screen === s ? "white" : "black" }}
              onMouseEnter={(e) => screen !== s && (e.target.style.backgroundColor = c.secondary)}
              onMouseLeave={(e) => screen !== s && (e.target.style.backgroundColor = "white")}
            >
              {stepLabels[i]}
            </button>
          ))}
        </div>

        <div className="flex border-2 border-black bg-white">
          <input className="w-40 px-3 py-2 outline-none" placeholder="Search..." />
          <button className="border-l-2 border-black px-3" style={{ backgroundColor: c.secondary }} aria-label="Search">
            <Icon name="search" size={20} />
          </button>
        </div>
      </nav>
    );
  }

  if (screen === "landing") {
    return (
      <div className="h-screen flex flex-col font-sans text-black overflow-hidden" style={{ backgroundColor: c.background }}>
        <div className="border-b-2 border-black" style={{ backgroundColor: c.header }}>
          <Navbar />
        </div>
        <div className="flex-1 flex flex-col items-center justify-center text-center relative overflow-hidden px-8" style={{ backgroundColor: c.header }}>
          {/* Theme Toggle Button - Top Left */}
          <button
            onClick={() => setTheme(theme === "multi" ? "simple" : "multi")}
            className="absolute top-4 left-4 z-50 border-2 border-black px-4 py-2 font-black shadow-[4px_4px_0px_#000] transition hover:-translate-y-0.5"
            style={{ backgroundColor: c.primary, color: c.white }}
          >
            {theme === "multi" ? "🎨 Multi" : "🟣 Simple"}
          </button>
          <div className="absolute left-8 top-8 h-24 w-24 rotate-12 border-2 border-black [clip-path:polygon(50%_0%,61%_35%,98%_35%,68%_57%,79%_91%,50%_70%,21%_91%,32%_57%,2%_35%,39%_35%)]" style={{ backgroundColor: c.secondary }} />
          <div className="absolute bottom-8 left-12 h-52 w-52 rounded-full border-2 border-black bg-white shadow-[inset_0_0_0_18px_#bd83dd,inset_0_0_0_36px_#fff,inset_0_0_0_54px_#bd83dd]" style={{ boxShadow: `inset 0 0 0 18px ${c.primary}, inset 0 0 0 36px #fff, inset 0 0 0 54px ${c.primary}` }} />
          <p className="relative text-3xl font-black [-webkit-text-stroke:1px_black]" style={{ color: c.accent2 }}>Your AI Reading Buddy</p>
          <h1 className="relative mx-auto mt-3 max-w-4xl text-6xl font-black leading-none text-white drop-shadow-[7px_7px_0px_#000] md:text-8xl">
            Brain Bling
          </h1>
          <p className="relative mx-auto mt-8 max-w-xl px-6 py-4 text-base font-semibold shadow-[6px_6px_0px_#000]" style={{ backgroundColor: c.secondary }}>
            Generate comprehension questions, verify answers, create distractors, and reveal smart hints using ML.
          </p>
          <button
            onClick={() => setScreen("input")}
            className="relative mt-10 inline-block rounded-full border-2 border-black px-10 py-4 text-2xl font-black shadow-[7px_7px_0px_#000] transition hover:-translate-y-1"
            style={{ backgroundColor: c.accent2, color: c.white }}
          >
            Start Quiz
          </button>
        </div>
      </div>
    );
  }

  if (screen === "input") {
    return (
      <div className="h-screen flex flex-col font-sans text-black overflow-hidden" style={{ backgroundColor: c.background }}>
        <div className="border-b-2 border-black" style={{ backgroundColor: c.header }}>
          <Navbar />
        </div>
        <main className="flex-1 flex items-center justify-center overflow-y-auto p-6">
          <section className="w-full max-w-3xl border-2 border-black p-8 shadow-[9px_9px_0px_#000]" style={{ backgroundColor: c.secondary }}>
            <div className="mb-4 flex items-center gap-3">
              <Icon name="upload" />
              <h2 className="text-3xl font-black">Step 1 - Article Input</h2>
            </div>

            {/* Model Selection - required before submit */}
            <div className="mb-5 border-2 border-black bg-white p-4 shadow-[4px_4px_0px_#000]">
              <p className="font-black mb-3">Choose Verifier Model for this session:</p>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => setModelType("bert")}
                  className={`border-2 border-black p-4 font-black shadow-[4px_4px_0px_#000] transition ${
                    modelType === "bert" ? "scale-105" : "hover:text-white"
                  }`}
                  style={{ backgroundColor: modelType === "bert" ? c.primary : "white", color: modelType === "bert" ? "white" : "black" }}
                  onMouseEnter={(e) => modelType !== "bert" && (e.target.style.backgroundColor = c.primary)}
                  onMouseLeave={(e) => modelType !== "bert" && (e.target.style.backgroundColor = "white")}
                >
                  <p>BERT (Neural)</p>
                  <p className="text-xs font-normal mt-1">RoBERTa fine-tuned on RACE<br/>Deep semantic understanding</p>
                </button>
                <button
                  onClick={() => setModelType("traditional")}
                  className={`border-2 border-black p-4 font-black shadow-[4px_4px_0px_#000] transition ${
                    modelType === "traditional" ? "scale-105" : ""
                  }`}
                  style={{ backgroundColor: modelType === "traditional" ? c.background : "white", color: modelType === "traditional" ? "black" : "black" }}
                  onMouseEnter={(e) => modelType !== "traditional" && (e.target.style.backgroundColor = c.background)}
                  onMouseLeave={(e) => modelType !== "traditional" && (e.target.style.backgroundColor = "white")}
                >
                  <p>Traditional (ML)</p>
                  <p className="text-xs font-normal mt-1">One-Hot + Cosine Similarity<br/>LR / SVM / Random Forest</p>
                </button>
              </div>
              {modelType && (
                <p className="mt-3 text-center font-bold text-sm border-2 border-black p-2"
                  style={{ backgroundColor: modelType === "bert" ? c.primary : c.background, color: modelType === "bert" ? "white" : "black" }}
                >
                  Selected: {modelType === "bert" ? "BERT (Neural)" : "Traditional (ML)"} - entire quiz will use this model
                </p>
              )}
            </div>

            <textarea
              value={article}
              onChange={(e) => setArticle(e.target.value)}
              placeholder="Paste your article here..."
              className="w-full h-48 border-2 border-black p-4 font-mono text-sm shadow-[4px_4px_0px_#000] focus:outline-none focus:ring-4 focus:ring-black"
            />
            <div className="mb-5 border-2 border-black p-4" style={{ backgroundColor: c.background }}>
              <p className="font-black mb-3">Article Stats</p>
              <p className="text-sm font-bold">Word Count: {article.split(" ").length}</p>
              <p className="text-sm font-bold">Character Count: {article.length}</p>
            </div>

            <div className="mt-4 flex flex-wrap gap-4">
              <button
                onClick={() => {
                  if (!modelType) { setMessage("Please select a model first."); return; }
                  handleSubmitArticle();
                }}
                className="border-2 border-black px-6 py-3 font-black shadow-[4px_4px_0px_#000] hover:-translate-y-0.5 transition"
                style={{ backgroundColor: modelType ? c.accent2 : "#d1d5db", cursor: modelType ? "pointer" : "not-allowed" }}
              >
                Submit Article
              </button>
              <button onClick={handleLoadSample} className="border-2 border-black bg-white px-6 py-3 font-black shadow-[4px_4px_0px_#000] hover:-translate-y-0.5 transition">
                Load Random RACE Sample
              </button>
            </div>
            <p className="mt-4 border-2 border-black bg-white p-3 font-bold">Status: {message}</p>
          </section>
        </main>
      </div>
    );
  }

  if (screen === "quiz") {
    return (
      <div className="h-screen flex flex-col font-sans text-black overflow-hidden" style={{ backgroundColor: c.background }}>
        <div className="border-b-2 border-black" style={{ backgroundColor: c.header }}>
          <Navbar />
        </div>
        <main className="flex-1 overflow-y-auto p-6 flex justify-center">
        <div className="w-full max-w-3xl">
          <section className="border-2 border-black bg-white p-8 shadow-[9px_9px_0px_#000]">
            <div className="mb-4 flex items-center gap-3">
              <Icon name="brain" />
              <h2 className="text-3xl font-black">Step 2 - Question and Answer Quiz</h2>
            </div>

            {article && (
              <div className="mb-5 border-2 border-black p-4" style={{ backgroundColor: c.background }}>
                <p className="text-sm font-bold mb-1">Passage:</p>
                <p className="text-base">{article}</p>
              </div>
            )}

            <div className="border-2 border-black p-5" style={{ backgroundColor: c.primary }}>
              <div className="flex items-center justify-between">
                <p className="text-xl font-black text-white">Question generated by Model A</p>
                <button
                  onClick={async () => {
                    const start = performance.now();
                    console.log("Starting hint generation at:", start);
                    try {
                      console.log("Calling hint API...");
                      const hintRes = await fetch(`${API_BASE_URL}/api/generate-hints`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ article, question, options: options.map(o => o.text), model_type: modelType || "traditional" })
                      });
                      const hintData = await hintRes.json();
                      console.log("Hint API response received");
                      // Store hints for display in popup
                      window.currentHints = hintData.hints;
                    } catch (error) {
                      console.error("Error generating hints:", error);
                      setMessage("Failed to generate hints. Please ensure the backend is running.");
                      window.currentHints = [];
                    }
                    const end = performance.now();
                    const hintTime = Math.round(end - start);
                    console.log("Hint time measured:", hintTime, "ms");
                    setInferenceTimes(prev => ({
                      ...prev,
                      hint: hintTime,
                      total: prev.verification + prev.distractor + hintTime,
                    }));
                    setShowHintPopup(true);
                  }}
                  className="border-2 border-black p-2 shadow-[2px_2px_0px_#000] hover:shadow-[3px_3px_0px_#000] transition"
                  style={{ backgroundColor: c.secondary }}
                  aria-label="Show hints"
                >
                  <Icon name="lightbulb" size={20} />
                </button>
              </div>
              <p className="mt-2 text-lg text-white">{question}</p>
            </div>

            <div className="mt-5 grid gap-3 md:grid-cols-2">
              {options.map((option) => (
                <button
                  key={option.id}
                  onClick={() => {
                    setSelected(option.id);
                    setChecked(false);
                    setMessage(`Selected option ${option.id}. Click Check Answer.`);
                  }}
                  className="border-2 border-black p-4 text-left font-bold shadow-[4px_4px_0px_#000] transition hover:-translate-y-0.5"
                  style={{ backgroundColor: selected === option.id ? c.accent2 : c.secondary }}
                >
                  {option.id}) {option.text}
                </button>
              ))}
            </div>


            <div className="mt-5 flex flex-wrap gap-4">
              <button onClick={handleCheckAnswer} className="border-2 border-black px-8 py-3 font-black shadow-[4px_4px_0px_#000] hover:-translate-y-0.5 transition" style={{ backgroundColor: c.background }}>
                Check Answer
              </button>
              {checked && (
                <button onClick={() => setScreen("dashboard")} className="border-2 border-black px-8 py-3 font-black shadow-[4px_4px_0px_#000] hover:-translate-y-0.5 transition" style={{ backgroundColor: c.accent2 }}>
                  View Results
                </button>
              )}
            </div>

            {checked && (
              <div className="mt-5 space-y-3">
                <div className="flex items-center gap-3 border-2 border-black p-4 text-white shadow-[4px_4px_0px_#000]"
                     style={{ backgroundColor: selected === correctAnswer ? "#22c55e" : c.accent }}>
                  {selected === correctAnswer ? <Icon name="check" /> : <Icon name="x" />}
                  <p className="font-black">
                    {selected === correctAnswer ? "Correct!" : `Incorrect. Correct answer: ${correctAnswer}`}
                  </p>
                </div>

                {/* Show result for selected model only */}
                {verifierResult && (
                  <div className="border-2 border-black p-4 shadow-[4px_4px_0px_#000]"
                       style={{ backgroundColor: modelType === "bert" ? c.primary : c.background }}>
                    <p className="font-black text-xs mb-1" style={{ color: modelType === "bert" ? "white" : "black" }}>
                      {modelType === "bert" ? "BERT (RoBERTa-RACE) - Neural" : "Traditional (One-Hot Cosine) - ML"}
                    </p>
                    <p className="text-4xl font-black" style={{ color: modelType === "bert" ? "white" : "black" }}>
                      {modelType === "bert"
                        ? (verifierResult.bert_confidence != null ? `${(verifierResult.bert_confidence * 100).toFixed(1)}% confidence` : "Not loaded")
                        : (verifierResult.trad_confidence != null ? `${(verifierResult.trad_confidence * 100).toFixed(1)}% confidence` : "Not loaded")}
                    </p>
                    <p className="text-xs mt-1" style={{ color: modelType === "bert" ? "white" : "black" }}>
                      {modelType === "bert" ? "Deep semantic understanding of passage" : "Word frequency cosine similarity"}
                    </p>
                  </div>
                )}
              </div>
            )}
          </section>
        </div>
        </main>

        {showHintPopup && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
            <div className="max-w-4xl w-full border-4 border-black shadow-[12px_12px_0px_#000]" style={{ backgroundColor: c.accent2 }}>
              <div className="flex items-center justify-between border-b-4 border-black p-4" style={{ backgroundColor: c.primary }}>
                <div className="flex items-center gap-3">
                  <Icon name="lightbulb" />
                  <h2 className="text-2xl font-black text-white">Hint Panel</h2>
                </div>
                <button
                  onClick={() => setShowHintPopup(false)}
                  className="border-2 border-black px-4 py-2 font-black text-white shadow-[3px_3px_0px_#000] hover:shadow-[4px_4px_0px_#000]"
                  style={{ backgroundColor: c.accent }}
                >
                  Close
                </button>
              </div>
              <div className="p-6">
                {window.currentHints && window.currentHints.length > 0 ? (
                  <div className="grid gap-4 md:grid-cols-3">
                    {window.currentHints.map((hint, i) => (
                      <div key={i} className="border-2 border-black p-4 shadow-[4px_4px_0px_#000]" style={{ backgroundColor: i % 2 === 1 ? c.secondary : "white" }}>
                        <h3 className="text-xl font-black">Hint {i + 1}</h3>
                        <p>{hint}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-center font-bold">No hints available. Please try generating hints again.</p>
                )}
              </div>
              <button
                onClick={() => {
                  setShowHintPopup(false);
                  setChecked(true);
                  setMessage(correctAnswer ? `✓ Correct answer revealed: ${correctAnswer}` : "Answer revealed!");
                }}
                className="mt-5 w-full border-2 border-black px-8 py-3 font-black shadow-[4px_4px_0px_#000] hover:shadow-[6px_6px_0px_#000] transition"
                style={{ backgroundColor: c.secondary }}
              >
                Reveal Answer
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  if (screen === "dashboard") {
    const allModels = [
      ...metrics.binary_metrics.map((m) => ({ ...m, type: "Base" })),
      ...metrics.ensemble_metrics.map((m) => ({ ...m, type: "Ensemble" })),
    ];
    const bestAccuracy = allModels.length ? Math.max(...allModels.map((m) => m.Accuracy)).toFixed(4) : "-";
    const bestF1 = allModels.length ? Math.max(...allModels.map((m) => m["Macro F1"])).toFixed(4) : "-";
    const bestAUC = allModels.length ? Math.max(...allModels.map((m) => m["ROC-AUC"])).toFixed(4) : "-";

    function getPerformanceBadge(models, current) {
      const accs = models.map((m) => Number(m.Accuracy));
      const max = Math.max(...accs);
      const min = Math.min(...accs);
      const avg = accs.reduce((a, b) => a + b, 0) / accs.length;
      const acc = Number(current.Accuracy);
      if (acc === max)
        return <span className="inline-block border-2 border-black px-2 py-0.5 text-xs font-black text-white shadow-[2px_2px_0px_#000]" style={{ backgroundColor: "#22c55e" }}>Best</span>;
      if (acc >= avg)
        return <span className="inline-block border-2 border-black px-2 py-0.5 text-xs font-black shadow-[2px_2px_0px_#000]" style={{ backgroundColor: c.secondary }}>Good</span>;
      if (acc === min)
        return <span className="inline-block border-2 border-black px-2 py-0.5 text-xs font-black text-white shadow-[2px_2px_0px_#000]" style={{ backgroundColor: c.accent }}>Weak</span>;
      return <span className="inline-block border-2 border-black px-2 py-0.5 text-xs font-black shadow-[2px_2px_0px_#000]" style={{ backgroundColor: c.accent2 }}>Fair</span>;
    }

    return (
      <div className="h-screen flex flex-col font-sans text-black overflow-hidden" style={{ backgroundColor: c.background }}>
        <div className="border-b-2 border-black" style={{ backgroundColor: c.header }}>
          <Navbar />
        </div>
        <main className="flex-1 overflow-y-auto p-6">
          <div className="mx-auto max-w-6xl">
            <div className="border-2 border-black bg-white p-6 shadow-[9px_9px_0px_#000]">
              <div className="mb-4 flex items-center gap-3">
                <Icon name="chart" />
                <h2 className="text-3xl font-black">Step 3 - Results and Analytics</h2>
              </div>

              {checked && (
                <div className={`mb-6 flex items-center gap-3 border-2 border-black p-4 text-white shadow-[4px_4px_0px_#000] ${
                  selected === correctAnswer ? "bg-[#22c55e]" : "bg-[#ff4d00]"
                }`}>
                  {selected === correctAnswer ? <Icon name="check" /> : <Icon name="x" />}
                  <p className="font-black text-xl">
                    {selected === correctAnswer ? "You got it correct!" : `You answered ${selected}. Correct answer was ${correctAnswer}.`}
                  </p>
                </div>
              )}

              <div className="grid gap-4 md:grid-cols-4 mb-6">
                {[
                  ["Total Models", allModels.length || "-"],
                  ["Best Accuracy", bestAccuracy],
                  ["Best Macro F1", bestF1],
                  ["Best ROC-AUC", bestAUC],
                ].map(([label, value]) => (
                  <div key={label} className="border-2 border-black p-5 text-center shadow-[4px_4px_0px_#000]" style={{ backgroundColor: c.background }}>
                    <p className="text-sm font-bold">{label}</p>
                    <p className="mt-2 text-3xl font-black">{value}</p>
                  </div>
                ))}
              </div>

              <h3 className="text-xl font-black mb-2">Session Inference Times</h3>
              <div className="grid gap-4 md:grid-cols-4 mb-6">
                {[
                  ["Verification", `${inferenceTimes.verification}ms`, c.accent2],
                  ["Distractor", `${inferenceTimes.distractor}ms`, c.secondary],
                  ["Hint", `${inferenceTimes.hint}ms`, c.background],
                  ["Total", `${inferenceTimes.total}ms`, c.primary],
                ].map(([label, value, bgColor]) => (
                  <div key={label} className="border-2 border-black p-5 text-center shadow-[4px_4px_0px_#000]" style={{ backgroundColor: bgColor }}>
                    <p className="text-sm font-black">{label}</p>
                    <p className="mt-2 text-3xl font-black">{value}</p>
                  </div>
                ))}
              </div>

              <h3 className="text-xl font-black mb-2">Base Models</h3>
              <div className="overflow-x-auto border-2 border-black mb-6">
                <table className="w-full text-left text-sm" style={{ backgroundColor: c.secondary }}>
                  <thead className="border-b-2 border-black" style={{ backgroundColor: c.primary }}>
                    <tr>
                      <th className="p-3 font-black">Model</th>
                      <th className="p-3 font-black">Accuracy</th>
                      <th className="p-3 font-black">Precision</th>
                      <th className="p-3 font-black">Recall</th>
                      <th className="p-3 font-black">Macro F1</th>
                      <th className="p-3 font-black">ROC-AUC</th>
                      <th className="p-3 font-black">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.binary_metrics.map((m, i) => (
                      <tr key={m.Model} className={`border-b-2 border-black ${i % 2 === 1 ? "bg-white" : ""}`}>
                        <td className="p-3 font-bold">{m.Model}</td>
                        <td className="p-3">{Number(m.Accuracy).toFixed(4)}</td>
                        <td className="p-3">{Number(m.Precision).toFixed(4)}</td>
                        <td className="p-3">{Number(m.Recall).toFixed(4)}</td>
                        <td className="p-3">{Number(m["Macro F1"]).toFixed(4)}</td>
                        <td className="p-3">{Number(m["ROC-AUC"]).toFixed(4)}</td>
                        <td className="p-3">{getPerformanceBadge(metrics.binary_metrics, m)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <h3 className="text-xl font-black mb-2">Neural vs Traditional - Comparison</h3>
              <div className="overflow-x-auto border-2 border-black mb-6">
                <table className="w-full bg-[#7cf5d2] text-left text-sm">
                  <thead className="border-b-2 border-black bg-[#ff4d00] text-white">
                    <tr>
                      <th className="p-3 font-black">Model</th>
                      <th className="p-3 font-black">Type</th>
                      <th className="p-3 font-black">Accuracy</th>
                      <th className="p-3 font-black">Macro F1</th>
                      <th className="p-3 font-black">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...metrics.binary_metrics.map(m => ({...m, Type: "Traditional"})),
                      ...metrics.ensemble_metrics.filter(m => m.Model === "Hard Voting").map(m => ({...m, Type: "Ensemble"})),
                      ...metrics.neural_metrics
                    ].map((m, i) => (
                      <tr key={m.Model + i} className={`border-b-2 border-black ${i % 2 === 1 ? "bg-white" : ""}`}>
                        <td className="p-3 font-bold">{m.Model}</td>
                        <td className="p-3">
                          <span className={`inline-block border-2 border-black px-2 py-0.5 text-xs font-black shadow-[2px_2px_0px_#000] ${
                            m.Type === "Neural" ? "bg-[#bd83dd] text-white" :
                            m.Type === "Ensemble" ? "bg-[#ffc736]" : "bg-white"
                          }`}>{m.Type}</span>
                        </td>
                        <td className="p-3 font-bold">
                          {m.Type === "Neural" && m.Accuracy === "Run eval"
                            ? <span className="text-gray-500 italic">Pending eval</span>
                            : typeof m.Accuracy === "number" ? m.Accuracy.toFixed(4) : m.Accuracy}
                        </td>
                        <td className="p-3">
                          {m.Type === "Neural" && m["Macro F1"] === "Run eval"
                            ? <span className="text-gray-500 italic">Pending eval</span>
                            : typeof m["Macro F1"] === "number" ? m["Macro F1"].toFixed(4) : m["Macro F1"]}
                        </td>
                        <td className="p-3">
                          {m.Type === "Neural"
                            ? <span className="inline-block border-2 border-black bg-[#bd83dd] px-2 py-0.5 text-xs font-black text-white shadow-[2px_2px_0px_#000]">Loaded</span>
                            : getPerformanceBadge(metrics.binary_metrics, m)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <h3 className="text-xl font-black mb-2">Ensemble Models</h3>
              <div className="overflow-x-auto border-2 border-black mb-6">
                <table className="w-full bg-[#ffc736] text-left text-sm">
                  <thead className="border-b-2 border-black bg-[#bd83dd]">
                    <tr>
                      <th className="p-3 font-black">Model</th>
                      <th className="p-3 font-black">Accuracy</th>
                      <th className="p-3 font-black">Precision</th>
                      <th className="p-3 font-black">Recall</th>
                      <th className="p-3 font-black">Macro F1</th>
                      <th className="p-3 font-black">ROC-AUC</th>
                      <th className="p-3 font-black">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.ensemble_metrics.map((m, i) => (
                      <tr key={m.Model} className={`border-b-2 border-black ${i % 2 === 1 ? "bg-white" : ""}`}>
                        <td className="p-3 font-bold">{m.Model}</td>
                        <td className="p-3">{Number(m.Accuracy).toFixed(4)}</td>
                        <td className="p-3">{Number(m.Precision).toFixed(4)}</td>
                        <td className="p-3">{Number(m.Recall).toFixed(4)}</td>
                        <td className="p-3">{Number(m["Macro F1"]).toFixed(4)}</td>
                        <td className="p-3">{Number(m["ROC-AUC"]).toFixed(4)}</td>
                        <td className="p-3">{getPerformanceBadge(metrics.ensemble_metrics, m)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <h3 className="text-xl font-black mb-2">NLG Evaluation — BLEU / ROUGE / METEOR</h3>
              {metrics.nlg_metrics ? (
                <div className="mb-6 space-y-4">
                  {[
                    { key: "question_generation", label: "Question Generation (Rule-based + ML Ranking)", color: "bg-[#7cf5d2]" },
                    { key: "distractors_traditional", label: "Distractor Generation — Traditional (RandomForest)", color: "bg-[#ffc736]" },
                    { key: "distractors_neural", label: "Distractor Generation — Neural (SentenceTransformer)", color: "bg-[#bd83dd]" },
                    { key: "hint_generation", label: "Hint Generation — Traditional (Logistic Regression)", color: "bg-[#ff8bd8]" },
                  ].map(({ key, label, color }) => {
                    const m = metrics.nlg_metrics[key];
                    if (!m) return (
                      <div key={key} className={`border-2 border-black ${color} p-4`}>
                        <p className="font-black mb-1">{label}</p>
                        <p className="text-sm text-gray-600">Not available (model not loaded)</p>
                      </div>
                    );
                    return (
                      <div key={key} className={`border-2 border-black ${color} p-4`}>
                        <p className="font-black mb-3">{label} <span className="text-xs font-normal text-gray-600">({m.samples} samples)</span></p>
                        <div className="grid grid-cols-5 gap-2">
                          {["BLEU","ROUGE-1","ROUGE-2","ROUGE-L","METEOR"].map(metric => (
                            <div key={metric} className="border-2 border-black bg-white p-2 shadow-[2px_2px_0px_#000] text-center">
                              <p className="text-xs font-black text-gray-500">{metric}</p>
                              <p className="text-lg font-black">{m[metric]}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="border-2 border-black bg-white p-4 mb-6 text-gray-500 font-bold">
                  NLG metrics unavailable — ensure dev.csv is in data/raw/
                </div>
              )}

              <h3 className="text-xl font-black mb-2 mt-4">Confusion Matrix — Option Selection (A/B/C/D)</h3>
              {metrics.confusion_matrices ? (
                <>
                  <div className="mb-3 flex gap-2 flex-wrap">
                    {Object.keys(metrics.confusion_matrices).map(m => (
                      <button key={m} onClick={() => setSelectedCMModel(m)}
                        className={`border-2 border-black px-3 py-1 text-sm font-black shadow-[2px_2px_0px_#000] transition ${selectedCMModel === m ? "bg-[#ff4d00] text-white" : "bg-white hover:bg-[#ffc736]"}`}
                      >{m}</button>
                    ))}
                  </div>
                  {metrics.confusion_matrices[selectedCMModel] && (
                    <div className="overflow-x-auto border-2 border-black mb-6">
                      <table className="text-center text-sm">
                        <thead>
                          <tr>
                            <th className="p-2 border-2 border-black bg-[#bd83dd] font-black">True↓ Pred→</th>
                            {metrics.confusion_matrices[selectedCMModel].labels.map(l => (
                              <th key={l} className="p-2 border-2 border-black bg-[#bd83dd] font-black">{l}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {metrics.confusion_matrices[selectedCMModel].matrix.map((row, ri) => {
                            const rowSum = row.reduce((a, b) => a + b, 0);
                            return (
                              <tr key={ri}>
                                <td className="p-2 border-2 border-black bg-[#ffc736] font-black">{metrics.confusion_matrices[selectedCMModel].labels[ri]}</td>
                                {row.map((val, ci) => {
                                  const intensity = rowSum > 0 ? val / rowSum : 0;
                                  const bg = ri === ci ? `rgba(34,197,94,${0.2 + intensity * 0.7})` : `rgba(255,77,0,${intensity * 0.6})`;
                                  return <td key={ci} className="p-3 border-2 border-black font-bold text-lg" style={{ backgroundColor: bg }}>{val}</td>;
                                })}
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                      <p className="text-xs text-gray-500 p-2">Rows = true label · Columns = predicted · Green diagonal = correct · Red = misclassified</p>
                    </div>
                  )}
                </>
              ) : (
                <div className="border-2 border-black bg-white p-4 mb-6 text-gray-500 font-bold">Confusion matrix unavailable — predicted_options_val.csv not found.</div>
              )}

              <div className="flex gap-3 flex-wrap">
                <button
                  onClick={() => {
                    const rows = [
                      ["Question","Options","Correct Answer","Your Answer","Verification (ms)","Distractor (ms)","Hint (ms)","Total (ms)"],
                      [question, options.join(" | "), correctAnswer, selected || "—", inferenceTimes.verification, inferenceTimes.distractor, inferenceTimes.hint, inferenceTimes.total]
                    ];
                    const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g,'""')}"`).join(",")).join("\n");
                    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
                    const a = document.createElement("a"); a.href = url; a.download = "session_results.csv"; a.click(); URL.revokeObjectURL(url);
                  }}
                  className="border-2 border-black bg-[#7cf5d2] px-8 py-3 font-black shadow-[4px_4px_0px_#000] hover:-translate-y-0.5 transition"
                >Export Session CSV</button>
                <button
                  onClick={() => { setScreen("landing"); setSelected(""); setChecked(false); setArticle(""); setMessage("Paste an article or load the sample to begin."); }}
                  className="border-2 border-black bg-[#ff8bd8] px-8 py-3 font-black shadow-[4px_4px_0px_#000] hover:-translate-y-0.5 transition"
                >Start Over</button>
              </div>
            </div>
          </div>
        </main>
      </div>
    );
  }
}
