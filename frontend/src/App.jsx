import React, { useState, useEffect } from "react";

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
  const [metrics, setMetrics] = useState({ binary_metrics: [], ensemble_metrics: [] });
  const [question, setQuestion] = useState("What is the main idea of the passage?");
  const [options, setOptions] = useState([
    { id: "A", text: "Mina wanted to avoid reading practice" },
    { id: "B", text: "Mina improved by practicing reading comprehension" },
    { id: "C", text: "The competition was about drawing pictures" },
    { id: "D", text: "Mina answered questions without reading the passage" },
  ]);
  const [correctAnswer, setCorrectAnswer] = useState("B");
  const [inferenceTimes, setInferenceTimes] = useState({
    verification: 0,
    distractor: 0,
    hint: 0,
    total: 0,
  });

  useEffect(() => {
    fetch("http://localhost:5000/metrics")
      .then((r) => r.json())
      .then((data) => setMetrics(data))
      .catch(() => {
        setMetrics({
          binary_metrics: [
            { Model: "Logistic Regression", Accuracy: 0.6725, Precision: 0.2947, Recall: 0.2224, "Macro F1": 0.5219, "ROC-AUC": 0.5387 },
            { Model: "SVM", Accuracy: 0.6741, Precision: 0.2962, Recall: 0.2206, "Macro F1": 0.5222, "ROC-AUC": 0.5388 },
            { Model: "Naive Bayes", Accuracy: 0.6218, Precision: 0.2525, Recall: 0.2617, "Macro F1": 0.5017, "ROC-AUC": 0.5034 },
            { Model: "Random Forest", Accuracy: 0.5932, Precision: 0.2751, Recall: 0.3837, "Macro F1": 0.5151, "ROC-AUC": 0.5364 },
          ],
          ensemble_metrics: [
            { Model: "Soft Voting", Accuracy: 0.5490, Precision: 0.2687, Recall: 0.4672, "Macro F1": 0.4992, "ROC-AUC": 0.5276 },
            { Model: "Hard Voting", Accuracy: 0.6809, Precision: 0.2948, Recall: 0.1985, "Macro F1": 0.5178, "ROC-AUC": 0.5276 },
            { Model: "Stacking", Accuracy: 0.6380, Precision: 0.2625, Recall: 0.2476, "Macro F1": 0.5079, "ROC-AUC": 0.5181 },
          ],
        });
      });
  }, []);

  async function handleLoadSample() {
    setMessage("Loading random sample from dataset...");
    try {
      const res = await fetch("http://localhost:5000/api/sample-article?" + Date.now(), {
        headers: { 
          "Cache-Control": "no-cache",
          "Pragma": "no-cache"
        }
      });
      const data = await res.json();
      
      // RESET ALL QUIZ STATE to prevent caching issues
      setArticle(data.article);
      setQuestion("What is the main idea of the passage?");
      setOptions([
        { id: "A", text: "" },
        { id: "B", text: "" },
        { id: "C", text: "" },
        { id: "D", text: "" }
      ]);
      setCorrectAnswer("");
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
      const sampleRes = await fetch("http://localhost:5000/api/sample-article?" + Date.now(), {
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
        const raceQuizRes = await fetch("http://localhost:5000/api/generate-race-quiz?" + Date.now(), {
          method: "POST",
          headers: { 
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
          },
          body: JSON.stringify({ 
            article: sampleData.article,
            question: sampleData.question,
            options: [sampleData.A.text, sampleData.B.text, sampleData.C.text, sampleData.D.text],
            correct_answer: sampleData.correct_answer
          })
        });
        quizData = await raceQuizRes.json();
        console.log("RACE quiz API response received - FRESH CALL");
      } else {
        console.log("Manual article detected, calling generate-quiz API...");
        // Use manual article workflow: generate question + 1 correct + 3 distractors
        const quizRes = await fetch("http://localhost:5000/api/generate-quiz", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ article })
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
      console.log("Calling check-answer API...");
      const res = await fetch("http://localhost:5000/api/check-answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          article,
          question,
          selected_option: selected,
          options: options.map(o => o.text),
          correct_answer: correctAnswer
        })
      });
      const data = await res.json();
      console.log("Check-answer API response received:", data);
      console.log("Correct answer from check-answer API:", data.correct_answer);
      setChecked(true);
      setMessage(`Model A verifier: ${data.explanation}`);
      setCorrectAnswer(data.correct_answer);
      console.log("Updated correctAnswer to:", data.correct_answer);
    } catch (error) {
      console.error("Error checking answer:", error);
      setChecked(true);
      setMessage("API unavailable. Using local verification.");
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
      <nav className="flex flex-wrap items-center justify-between gap-4 border-b-2 border-black bg-[#bd83dd] px-6 py-4">
        <button
          onClick={() => setScreen("landing")}
          className="border-2 border-black bg-[#ffc736] px-5 py-2 text-3xl font-black tracking-tight shadow-[4px_4px_0px_#000]"
        >
          Brain Bling
        </button>

        <div className="flex gap-2">
          {steps.map((s, i) => (
            <button
              key={s}
              onClick={() => setScreen(s)}
              className={`border-2 border-black px-4 py-2 text-sm font-bold shadow-[3px_3px_0px_#000] transition ${
                screen === s ? "bg-[#ff4d00] text-white" : "bg-white hover:bg-[#ffc736]"
              }`}
            >
              {stepLabels[i]}
            </button>
          ))}
        </div>

        <div className="flex border-2 border-black bg-white">
          <input className="w-40 px-3 py-2 outline-none" placeholder="Search..." />
          <button className="border-l-2 border-black bg-[#ffc736] px-3" aria-label="Search">
            <Icon name="search" size={20} />
          </button>
        </div>
      </nav>
    );
  }

  if (screen === "landing") {
    return (
      <div className="h-screen flex flex-col bg-[#7cf5d2] font-sans text-black overflow-hidden">
        <div className="bg-[#ff4d00] border-b-2 border-black">
          <Navbar />
        </div>
        <div className="flex-1 flex flex-col items-center justify-center text-center relative overflow-hidden px-8 bg-[#ff4d00]">
          <div className="absolute left-8 top-8 h-24 w-24 rotate-12 border-2 border-black bg-[#ffc736] [clip-path:polygon(50%_0%,61%_35%,98%_35%,68%_57%,79%_91%,50%_70%,21%_91%,32%_57%,2%_35%,39%_35%)]" />
          <div className="absolute bottom-8 left-12 h-52 w-52 rounded-full border-2 border-black bg-white shadow-[inset_0_0_0_18px_#bd83dd,inset_0_0_0_36px_#fff,inset_0_0_0_54px_#bd83dd]" />
          <p className="relative text-3xl font-black text-[#ff8bd8] [-webkit-text-stroke:1px_black]">Your AI Reading Buddy</p>
          <h1 className="relative mx-auto mt-3 max-w-4xl text-6xl font-black leading-none text-white drop-shadow-[7px_7px_0px_#000] md:text-8xl">
            Brain Bling
          </h1>
          <p className="relative mx-auto mt-8 max-w-xl bg-[#ffc736] px-6 py-4 text-base font-semibold shadow-[6px_6px_0px_#000]">
            Generate comprehension questions, verify answers, create distractors, and reveal smart hints using ML.
          </p>
          <button
            onClick={() => setScreen("input")}
            className="relative mt-10 inline-block rounded-full border-2 border-black bg-[#ff8bd8] px-10 py-4 text-2xl font-black shadow-[7px_7px_0px_#000] transition hover:-translate-y-1"
          >
            Start Quiz
          </button>
        </div>
      </div>
    );
  }

  if (screen === "input") {
    return (
      <div className="h-screen flex flex-col bg-[#7cf5d2] font-sans text-black overflow-hidden">
        <div className="bg-[#ff4d00] border-b-2 border-black">
          <Navbar />
        </div>
        <main className="flex-1 flex items-center justify-center overflow-y-auto p-6">
          <section className="w-full max-w-3xl border-2 border-black bg-[#ffc736] p-8 shadow-[9px_9px_0px_#000]">
            <div className="mb-4 flex items-center gap-3">
              <Icon name="upload" />
              <h2 className="text-3xl font-black">Step 1 — Article Input</h2>
            </div>
            <textarea
              value={article}
              onChange={(event) => setArticle(event.target.value)}
              className="min-h-48 w-full border-2 border-black bg-white p-4 text-base outline-none"
              placeholder="Paste reading passage here..."
            />
            <div className="mt-4 flex flex-wrap gap-4">
              <button onClick={handleSubmitArticle} className="border-2 border-black bg-[#ff8bd8] px-6 py-3 font-black shadow-[4px_4px_0px_#000] hover:-translate-y-0.5 transition">
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
      <div className="h-screen flex flex-col bg-[#7cf5d2] font-sans text-black overflow-hidden">
        <div className="bg-[#ff4d00] border-b-2 border-black">
          <Navbar />
        </div>
        <main className="flex-1 overflow-y-auto p-6 flex justify-center">
        <div className="w-full max-w-3xl">
          <section className="border-2 border-black bg-white p-8 shadow-[9px_9px_0px_#000]">
            <div className="mb-4 flex items-center gap-3">
              <Icon name="brain" />
              <h2 className="text-3xl font-black">Step 2 — Question & Answer Quiz</h2>
            </div>

            {article && (
              <div className="mb-5 border-2 border-black bg-[#7cf5d2] p-4">
                <p className="text-sm font-bold mb-1">Passage:</p>
                <p className="text-base">{article}</p>
              </div>
            )}

            <div className="border-2 border-black bg-[#bd83dd] p-5">
              <div className="flex items-center justify-between">
                <p className="text-xl font-black">Question generated by Model A</p>
                <button
                  onClick={async () => {
                    const start = performance.now();
                    console.log("Starting hint generation at:", start);
                    try {
                      console.log("Calling hint API...");
                      const hintRes = await fetch("http://localhost:5000/api/generate-hints", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ article, question, options: options.map(o => o.text) })
                      });
                      const hintData = await hintRes.json();
                      console.log("Hint API response received");
                      // Store hints for display in popup
                      window.currentHints = hintData.hints;
                    } catch (error) {
                      console.error("Error generating hints:", error);
                      window.currentHints = [
                        "Hint 1: Look for key information in the passage",
                        "Hint 2: Focus on the main topic",
                        "Hint 3: Consider the context"
                      ];
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
                  className="border-2 border-black bg-[#ffc736] p-2 shadow-[2px_2px_0px_#000] hover:shadow-[3px_3px_0px_#000] transition"
                  aria-label="Show hints"
                >
                  <Icon name="lightbulb" size={20} />
                </button>
              </div>
              <p className="mt-2 text-lg">{question}</p>
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
                  className={`border-2 border-black p-4 text-left font-bold shadow-[4px_4px_0px_#000] transition hover:-translate-y-0.5 ${
                    selected === option.id ? "bg-[#ff8bd8]" : "bg-[#ffc736]"
                  }`}
                >
                  {option.id}) {option.text}
                </button>
              ))}
            </div>

            <div className="mt-5 flex flex-wrap gap-4">
              <button onClick={handleCheckAnswer} className="border-2 border-black bg-[#7cf5d2] px-8 py-3 font-black shadow-[4px_4px_0px_#000] hover:-translate-y-0.5 transition">
                Check Answer
              </button>
              {checked && (
                <button onClick={() => setScreen("dashboard")} className="border-2 border-black bg-[#ff8bd8] px-8 py-3 font-black shadow-[4px_4px_0px_#000] hover:-translate-y-0.5 transition">
                  View Results
                </button>
              )}
            </div>

            {checked && (
              <div className={`mt-5 flex items-center gap-3 border-2 border-black p-4 text-white shadow-[4px_4px_0px_#000] ${
                selected === correctAnswer ? "bg-[#22c55e]" : "bg-[#ff4d00]"
              }`}>
                {selected === correctAnswer ? <Icon name="check" /> : <Icon name="x" />}
                <p className="font-black">
                  {selected === correctAnswer ? "Correct!" : `Incorrect. Correct answer: ${correctAnswer}`} Model A verifier result shown here.
                </p>
              </div>
            )}
          </section>
        </div>
        </main>

        {showHintPopup && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
            <div className="max-w-4xl w-full border-4 border-black bg-[#ff8bd8] shadow-[12px_12px_0px_#000]">
              <div className="flex items-center justify-between border-b-4 border-black bg-[#bd83dd] p-4">
                <div className="flex items-center gap-3">
                  <Icon name="lightbulb" />
                  <h2 className="text-2xl font-black">Hint Panel</h2>
                </div>
                <button
                  onClick={() => setShowHintPopup(false)}
                  className="border-2 border-black bg-[#ff4d00] px-4 py-2 font-black text-white shadow-[3px_3px_0px_#000] hover:shadow-[4px_4px_0px_#000]"
                >
                  Close
                </button>
              </div>
              <div className="p-6">
                <div className="grid gap-4 md:grid-cols-3">
                  {(window.currentHints || [
                    "Hint 1: Look for key information in the passage",
                    "Hint 2: Focus on the main topic",
                    "Hint 3: Consider the context"
                  ]).map((hint, i) => (
                    <div key={i} className={`border-2 border-black ${i % 2 === 1 ? "bg-[#ffc736]" : "bg-white"} p-4 shadow-[4px_4px_0px_#000]`}>
                      <h3 className="text-xl font-black">Hint {i + 1}</h3>
                      <p>{hint}</p>
                    </div>
                  ))}
                </div>
              </div>
              <button className="mt-5 w-full border-2 border-black bg-[#ffc736] px-8 py-3 font-black shadow-[4px_4px_0px_#000]">
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
        return <span className="inline-block border-2 border-black bg-[#22c55e] px-2 py-0.5 text-xs font-black text-white shadow-[2px_2px_0px_#000]">Best</span>;
      if (acc >= avg)
        return <span className="inline-block border-2 border-black bg-[#ffc736] px-2 py-0.5 text-xs font-black shadow-[2px_2px_0px_#000]">Good</span>;
      if (acc === min)
        return <span className="inline-block border-2 border-black bg-[#ff4d00] px-2 py-0.5 text-xs font-black text-white shadow-[2px_2px_0px_#000]">Weak</span>;
      return <span className="inline-block border-2 border-black bg-[#ff8bd8] px-2 py-0.5 text-xs font-black shadow-[2px_2px_0px_#000]">Fair</span>;
    }

    return (
      <div className="h-screen flex flex-col bg-[#7cf5d2] font-sans text-black overflow-hidden">
        <div className="bg-[#ff4d00] border-b-2 border-black">
          <Navbar />
        </div>
        <main className="flex-1 overflow-y-auto p-6">
          <div className="mx-auto max-w-6xl">
            <div className="border-2 border-black bg-white p-6 shadow-[9px_9px_0px_#000]">
              <div className="mb-4 flex items-center gap-3">
                <Icon name="chart" />
                <h2 className="text-3xl font-black">Step 3 — Results & Analytics</h2>
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
                  <div key={label} className="border-2 border-black bg-[#7cf5d2] p-5 text-center shadow-[4px_4px_0px_#000]">
                    <p className="text-sm font-bold">{label}</p>
                    <p className="mt-2 text-3xl font-black">{value}</p>
                  </div>
                ))}
              </div>

              <h3 className="text-xl font-black mb-2">Session Inference Times</h3>
              <div className="grid gap-4 md:grid-cols-4 mb-6">
                {[
                  ["Verification", `${inferenceTimes.verification}ms`, "bg-[#ff8bd8]"],
                  ["Distractor", `${inferenceTimes.distractor}ms`, "bg-[#ffc736]"],
                  ["Hint", `${inferenceTimes.hint}ms`, "bg-[#7cf5d2]"],
                  ["Total", `${inferenceTimes.total}ms`, "bg-[#bd83dd]"],
                ].map(([label, value, bgColor]) => (
                  <div key={label} className={`border-2 border-black ${bgColor} p-5 text-center shadow-[4px_4px_0px_#000]`}>
                    <p className="text-sm font-black">{label}</p>
                    <p className="mt-2 text-3xl font-black">{value}</p>
                  </div>
                ))}
              </div>

              <h3 className="text-xl font-black mb-2">Base Models</h3>
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

              <button
                onClick={() => { setScreen("landing"); setSelected(""); setChecked(false); setArticle(""); setMessage("Paste an article or load the sample to begin."); }}
                className="border-2 border-black bg-[#ff8bd8] px-8 py-3 font-black shadow-[4px_4px_0px_#000] hover:-translate-y-0.5 transition"
              >
                Start Over
              </button>
            </div>
          </div>
        </main>
      </div>
    );
  }
}
