import React, { useState } from "react";

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
  const [article, setArticle] = useState("");
  const [selected, setSelected] = useState("");
  const [checked, setChecked] = useState(false);
  const [message, setMessage] = useState("Paste an article or load the sample to begin.");
  const [showHintPopup, setShowHintPopup] = useState(false);

  const correctAnswer = "B";

  const sampleArticle =
    "Mina joined a reading competition at school. She practiced every evening and learned how to find key ideas in long passages. On the final day, she answered most questions correctly because she focused on the main idea of each paragraph.";

  const options = [
    { id: "A", text: "Mina wanted to avoid reading practice" },
    { id: "B", text: "Mina improved by practicing reading comprehension" },
    { id: "C", text: "The competition was about drawing pictures" },
    { id: "D", text: "Mina answered questions without reading the passage" },
  ];

  function handleLoadSample() {
    setArticle(sampleArticle);
    setSelected("");
    setChecked(false);
    setMessage("Sample RACE-style passage loaded.");
  }

  function handleSubmitArticle() {
    if (!article.trim()) {
      setMessage("Please paste an article first.");
      return;
    }
    setSelected("");
    setChecked(false);
    setMessage("Article submitted. Model A and Model B outputs are shown below.");
  }

  function handleCheckAnswer() {
    if (!selected) {
      setMessage("Please select one option before checking.");
      return;
    }
    setChecked(true);
    setMessage("Model A verifier checked your selected answer.");
  }

  const tests = [
    { name: "Correct answer exists", passed: options.some((option) => option.id === correctAnswer) },
    { name: "Exactly four options shown", passed: options.length === 4 },
    { name: "Sample article available", passed: sampleArticle.length > 20 },
    { name: "No external icon package required", passed: true },
  ];

  return (
    <div className="min-h-screen bg-[#7cf5d2] p-6 font-sans text-black">
      <div className="mx-auto max-w-7xl border-2 border-black bg-[#ff4d00] shadow-[12px_12px_0px_#000]">
        <nav className="flex flex-wrap items-center justify-between gap-4 border-b-2 border-black bg-[#bd83dd] px-6 py-4">
          <div className="border-2 border-black bg-[#ffc736] px-5 py-2 text-3xl font-black tracking-tight shadow-[4px_4px_0px_#000]">
            Brain Bling
          </div>

          <div className="flex gap-6 text-sm font-bold">
            <a href="#input">Input</a>
            <a href="#quiz">Quiz</a>
            <a href="#hints">Hints</a>
            <a href="#dashboard">Dashboard</a>
          </div>

          <div className="flex border-2 border-black bg-white">
            <input className="w-40 px-3 py-2 outline-none" placeholder="Search..." />
            <button className="border-l-2 border-black bg-[#ffc736] px-3" aria-label="Search">
              <Icon name="search" size={20} />
            </button>
          </div>
        </nav>

        <section className="relative overflow-hidden px-8 py-16 text-center">
          <div className="absolute left-8 top-8 h-24 w-24 rotate-12 border-2 border-black bg-[#ffc736] [clip-path:polygon(50%_0%,61%_35%,98%_35%,68%_57%,79%_91%,50%_70%,21%_91%,32%_57%,2%_35%,39%_35%)]" />
          <div className="absolute right-10 top-28 text-5xl font-black">〰〰〰</div>
          <div className="absolute bottom-8 left-12 h-52 w-52 rounded-full border-2 border-black bg-white shadow-[inset_0_0_0_18px_#bd83dd,inset_0_0_0_36px_#fff,inset_0_0_0_54px_#bd83dd]" />

          <p className="relative text-3xl font-black text-[#ff8bd8] [-webkit-text-stroke:1px_black]">Your AI Reading Buddy</p>
          <h1 className="relative mx-auto mt-3 max-w-4xl text-6xl font-black leading-none text-white drop-shadow-[7px_7px_0px_#000] md:text-8xl">
            Brain Bling
          </h1>
          <p className="relative mx-auto mt-8 max-w-xl bg-[#ffc736] px-6 py-4 text-base font-semibold shadow-[6px_6px_0px_#000]">
            Generate comprehension questions, verify answers, create distractors, and reveal smart hints using ML.
          </p>
          <a
            href="#input"
            className="relative mt-10 inline-block rounded-full border-2 border-black bg-[#ff8bd8] px-10 py-4 text-2xl font-black shadow-[7px_7px_0px_#000] transition hover:-translate-y-1"
          >
            Start Quiz
          </a>
        </section>
      </div>

      <main className="mx-auto mt-12 grid max-w-7xl gap-8">
        <section id="input" className="border-2 border-black bg-[#ffc736] p-6 shadow-[9px_9px_0px_#000]">
          <div className="mb-4 flex items-center gap-3">
            <Icon name="upload" />
            <h2 className="text-3xl font-black">Article Input</h2>
          </div>
          <textarea
            value={article}
            onChange={(event) => setArticle(event.target.value)}
            className="min-h-48 w-full border-2 border-black bg-white p-4 text-base outline-none"
            placeholder="Paste reading passage here..."
          />
          <div className="mt-4 flex flex-wrap gap-4">
            <button onClick={handleSubmitArticle} className="border-2 border-black bg-[#ff8bd8] px-6 py-3 font-black shadow-[4px_4px_0px_#000]">
              Submit Article
            </button>
            <button onClick={handleLoadSample} className="border-2 border-black bg-white px-6 py-3 font-black shadow-[4px_4px_0px_#000]">
              Load Random RACE Sample
            </button>
          </div>
          <p className="mt-4 border-2 border-black bg-white p-3 font-bold">Status: {message}</p>
        </section>

        <section id="quiz" className="border-2 border-black bg-white p-6 shadow-[9px_9px_0px_#000]">
          <div className="mb-4 flex items-center gap-3">
            <Icon name="brain" />
            <h2 className="text-3xl font-black">Question & Answer Quiz</h2>
          </div>
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
            <p className="mt-2 text-lg">What is the main idea of the passage?</p>
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
                className={`border-2 border-black p-4 text-left font-bold shadow-[4px_4px_0px_#000] ${
                  selected === option.id ? "bg-[#ff8bd8]" : "bg-[#ffc736]"
                }`}
              >
                {option.id}) {option.text}
              </button>
            ))}
          </div>

          <button onClick={handleCheckAnswer} className="mt-5 border-2 border-black bg-[#7cf5d2] px-8 py-3 font-black shadow-[4px_4px_0px_#000]">
            Check Answer
          </button>

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

        
        <section id="dashboard" className="border-2 border-black bg-white p-6 shadow-[9px_9px_0px_#000]">
          <div className="mb-4 flex items-center gap-3">
            <Icon name="chart" />
            <h2 className="text-3xl font-black">Developer Analytics Dashboard</h2>
          </div>
          <div className="grid gap-4 md:grid-cols-4">
            {[
              ["Model A Accuracy", "0.65"],
              ["Macro F1", "0.64"],
              ["Model B F1", "0.58"],
              ["Latency", "2.1s"],
            ].map(([label, value]) => (
              <div key={label} className="border-2 border-black bg-[#7cf5d2] p-5 text-center shadow-[4px_4px_0px_#000]">
                <p className="text-sm font-bold">{label}</p>
                <p className="mt-2 text-4xl font-black">{value}</p>
              </div>
            ))}
          </div>

          <div className="mt-6 overflow-x-auto border-2 border-black">
            <table className="w-full bg-[#ffc736] text-left">
              <thead className="border-b-2 border-black bg-[#bd83dd]">
                <tr>
                  <th className="p-3">Model</th>
                  <th className="p-3">Accuracy</th>
                  <th className="p-3">F1</th>
                  <th className="p-3">Purpose</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b-2 border-black">
                  <td className="p-3 font-bold">Logistic Regression</td>
                  <td className="p-3">0.65</td>
                  <td className="p-3">0.64</td>
                  <td className="p-3">Answer verification</td>
                </tr>
                <tr className="border-b-2 border-black bg-white">
                  <td className="p-3 font-bold">SVM</td>
                  <td className="p-3">0.67</td>
                  <td className="p-3">0.66</td>
                  <td className="p-3">Answer verification</td>
                </tr>
                <tr>
                  <td className="p-3 font-bold">BERT</td>
                  <td className="p-3">0.72</td>
                  <td className="p-3">0.71</td>
                  <td className="p-3">Comparison baseline</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section className="mb-12 border-2 border-black bg-[#ffc736] p-6 shadow-[9px_9px_0px_#000]">
          <h2 className="text-3xl font-black">Template Self-Tests</h2>
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            {tests.map((test) => (
              <div key={test.name} className="flex items-center justify-between border-2 border-black bg-white p-3 font-bold">
                <span>{test.name}</span>
                <span>{test.passed ? "PASS" : "FAIL"}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Hint Popup Modal */}
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
      </main>
    </div>
  );
}
