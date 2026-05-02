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
      const res = await fetch("http://localhost:5000/api/sample-article");
      const data = await res.json();
      setArticle(data.article);
      setQuestion(data.question);
      setOptions(data.options);
      setCorrectAnswer(data.correct_answer);
      setSelected("");
      setChecked(false);
      setMessage("Random RACE sample loaded from dataset.");
    } catch {
      setMessage("API unavailable. Please start the backend server.");
    }
  }

  function handleSubmitArticle() {
    if (!article.trim()) {
      setMessage("Please paste an article first.");
      return;
    }
    setSelected("");
    setChecked(false);
    setScreen("quiz");
  }

  function handleCheckAnswer() {
    if (!selected) {
      setMessage("Please select one option before checking.");
      return;
    }
    setChecked(true);
    setMessage("Model A verifier checked your selected answer.");
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
                  onClick={() => setShowHintPopup(true)}
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
                  <div className="border-2 border-black bg-white p-4 shadow-[4px_4px_0px_#000]">
                    <h3 className="text-xl font-black">Hint 1</h3>
                    <p>Look for the repeated learning action in the passage.</p>
                  </div>
                  <div className="border-2 border-black bg-[#ffc736] p-4 shadow-[4px_4px_0px_#000]">
                    <h3 className="text-xl font-black">Hint 2</h3>
                    <p>The passage focuses on reading practice before a competition.</p>
                  </div>
                  <div className="border-2 border-black bg-white p-4 shadow-[4px_4px_0px_#000]">
                    <h3 className="text-xl font-black">Hint 3</h3>
                    <p>The best option mentions improvement through reading comprehension practice.</p>
                  </div>
                </div>
                <button className="mt-5 w-full border-2 border-black bg-[#ffc736] px-8 py-3 font-black shadow-[4px_4px_0px_#000]">
                  Reveal Answer
                </button>
              </div>
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
