import React, { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity, BookOpen, CheckCheck, ChevronRight, CircleHelp, ClipboardCheck,
  FileQuestion, Gauge, GraduationCap, LayoutDashboard, LoaderCircle, ScanText,
  Settings2, ShieldCheck, Sparkles, Wifi, WifiOff
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const navItems = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "questions", label: "Question Generator", icon: FileQuestion },
  { id: "rag", label: "Ask Study Material", icon: BookOpen },
  { id: "evaluation", label: "Answer Evaluation", icon: ClipboardCheck },
  { id: "batch", label: "Batch Evaluation", icon: CheckCheck },
  { id: "plagiarism", label: "Plagiarism Check", icon: ScanText },
];

const pageInfo = {
  dashboard: { eyebrow: "Overview", title: "Good to see you." },
  questions: { eyebrow: "Question Generator", title: "Generate questions." },
  rag: { eyebrow: "Study Assistant", title: "Ask your study material." },
  evaluation: { eyebrow: "Answer Evaluation", title: "Evaluate an answer." },
  batch: { eyebrow: "Batch Evaluation", title: "Evaluate submissions in bulk." },
  plagiarism: { eyebrow: "Plagiarism Check", title: "Check text similarity." },
};

const defaultQuestionBody = {
  source_text:
    "Photosynthesis is the process by which green plants use sunlight, carbon dioxide, and water to produce glucose and oxygen. Chlorophyll absorbs light energy. Photosynthesis occurs mainly in the chloroplasts of plant cells.",
  num_questions: 3,
  difficulty: "medium",
  question_types: ["mcq"],
  total_marks: 3,
  subject: "Biology",
  instructions: "Use only the supplied source material.",
};

const starterJson = {
  rag: {
    question: "Explain photosynthesis in simple terms.",
    context:
      "Photosynthesis uses sunlight, water and carbon dioxide to produce glucose and oxygen.",
  },
  evaluation: {
    question: "What is photosynthesis?",
    student_answer: "Plants use sunlight to make food.",
    reference_answer:
      "Green plants use light energy to convert carbon dioxide and water into glucose and oxygen.",
    max_marks: 5,
  },
  batch: {
    items: [
      {
        question: "What is photosynthesis?",
        student_answer: "Plants use sunlight to make food.",
        reference_answer:
          "Plants use light energy to convert carbon dioxide and water into glucose and oxygen.",
        max_marks: 5,
      },
    ],
  },
  plagiarism: {
    texts: [
      "Photosynthesis allows green plants to convert light energy into chemical energy.",
      "Green plants use light to convert energy into a chemical form through photosynthesis.",
    ],
  },
};

const endpoints = {
  rag: "/api/rag/ask",
  evaluation: "/api/evaluation/evaluate",
  batch: "/api/evaluation/evaluate-batch",
  plagiarism: "/api/plagiarism/check",
};

function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [health, setHealth] = useState({ loading: true, data: null, error: "" });
  const [questionForm, setQuestionForm] = useState(defaultQuestionBody);
  const [showQuestionJson, setShowQuestionJson] = useState(false);
  const [questionJson, setQuestionJson] = useState(
    JSON.stringify(defaultQuestionBody, null, 2)
  );
  const [jsonBodies, setJsonBodies] = useState(() =>
    Object.fromEntries(
      Object.entries(starterJson).map(([key, value]) => [
        key,
        JSON.stringify(value, null, 2),
      ])
    )
  );
  const [results, setResults] = useState({});
  const [busy, setBusy] = useState("");

  const apiRequest = useCallback(async (path, options = {}) => {
    let response;
    try {
      response = await fetch(`${API_BASE}${path}`, options);
    } catch {
      throw new Error(
        `Cannot reach ${API_BASE}. Start your FastAPI backend and check CORS settings.`
      );
    }
    const raw = await response.text();
    let data;
    try {
      data = raw ? JSON.parse(raw) : {};
    } catch {
      data = raw;
    }
    if (!response.ok) {
      const error = new Error(`HTTP ${response.status}`);
      error.payload = data;
      throw error;
    }
    return data;
  }, []);

  const refreshHealth = useCallback(async () => {
    setHealth({ loading: true, data: null, error: "" });
    try {
      const data = await apiRequest("/health");
      setHealth({ loading: false, data, error: "" });
    } catch (error) {
      setHealth({ loading: false, data: null, error: error.message });
    }
  }, [apiRequest]);

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  function setQuestionField(field, value) {
    setQuestionForm((current) => {
      const next = { ...current, [field]: value };
      setQuestionJson(JSON.stringify(next, null, 2));
      return next;
    });
  }

  async function runRequest(key, path, body, resultKey = key) {
    setBusy(key);
    setResults((current) => ({ ...current, [resultKey]: null }));
    try {
      const data = await apiRequest(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setResults((current) => ({
        ...current,
        [resultKey]: { ok: true, data },
      }));
    } catch (error) {
      setResults((current) => ({
        ...current,
        [resultKey]: {
          ok: false,
          data: error.payload || { error: error.message },
        },
      }));
    } finally {
      setBusy("");
    }
  }

  function runJsonTool(key) {
    try {
      const body = JSON.parse(jsonBodies[key]);
      runRequest(key, endpoints[key], body);
    } catch {
      setResults((current) => ({
        ...current,
        [key]: {
          ok: false,
          data: { error: "Invalid JSON. Check commas, quotes and braces." },
        },
      }));
    }
  }

  const info = pageInfo[activePage];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><GraduationCap size={22} /></div>
          <div><strong>ScoreMatrix</strong><span>TEACHER WORKSPACE</span></div>
        </div>
        <div className="nav-caption">WORKSPACE</div>
        <nav className="nav-list" aria-label="Main navigation">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${activePage === id ? "active" : ""}`}
              onClick={() => setActivePage(id)}
            >
              <Icon size={18} strokeWidth={1.9} />
              <span>{label}</span>
              {activePage === id && <ChevronRight className="nav-chevron" size={15} />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-bottom-icon"><Settings2 size={17} /></div>
          <div><b>Local development</b><span>{API_BASE}</span></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">SCOREMATRIX <span>/</span> {info.eyebrow}</div>
            <h1>{info.title}</h1>
            <p className="page-subtitle">A smarter workspace for assessment and feedback.</p>
          </div>
          <div className={`connection-pill ${health.data ? "connected" : health.loading ? "" : "disconnected"}`}>
            {health.loading ? <LoaderCircle size={15} className="spin" /> : health.data ? <Wifi size={15} /> : <WifiOff size={15} />}
            {health.loading ? "Checking API" : health.data ? "Backend connected" : "Backend offline"}
          </div>
        </header>

        {activePage === "dashboard" && (
          <Dashboard health={health} refreshHealth={refreshHealth} go={setActivePage} />
        )}

        {activePage === "questions" && (
          <div className="content-grid">
            <section className="panel span-8">
              <PanelHeading icon={Sparkles} title="Question Generator" subtitle="Turn source material into structured questions." />
              <div className="field">
                <label>Source material</label>
                <textarea className="source-area" value={questionForm.source_text} onChange={(e) => setQuestionField("source_text", e.target.value)} />
              </div>
              <div className="form-row">
                <div className="field"><label>Number of questions</label><input type="number" min="1" max="50" value={questionForm.num_questions} onChange={(e) => setQuestionField("num_questions", Number(e.target.value))} /></div>
                <div className="field"><label>Difficulty</label><select value={questionForm.difficulty} onChange={(e) => setQuestionField("difficulty", e.target.value)}><option>easy</option><option>medium</option><option>hard</option></select></div>
              </div>
              <div className="form-row">
                <div className="field"><label>Question type</label><select value={questionForm.question_types[0]} onChange={(e) => setQuestionField("question_types", [e.target.value])}><option value="mcq">Multiple choice (MCQ)</option><option value="short_answer">Short answer</option><option value="true_false">True / False</option></select></div>
                <div className="field"><label>Total marks</label><input type="number" min="1" value={questionForm.total_marks} onChange={(e) => setQuestionField("total_marks", Number(e.target.value))} /></div>
              </div>
              <div className="form-row">
                <div className="field"><label>Subject</label><input value={questionForm.subject} onChange={(e) => setQuestionField("subject", e.target.value)} /></div>
                <div className="field"><label>Instructions</label><input value={questionForm.instructions} onChange={(e) => setQuestionField("instructions", e.target.value)} /></div>
              </div>
              <div className="button-row">
                <button className="button primary" disabled={busy === "questions"} onClick={() => {
                  let body = questionForm;
                  if (showQuestionJson) {
                    try { body = JSON.parse(questionJson); }
                    catch { setResults((r) => ({ ...r, questions: { ok: false, data: { error: "Invalid JSON." } } })); return; }
                  }
                  runRequest("questions", "/api/questions/generate", body);
                }}>{busy === "questions" ? <LoaderCircle size={16} className="spin" /> : <Sparkles size={16} />} Generate questions</button>
                <button className="button secondary" onClick={() => setShowQuestionJson((v) => !v)}>{showQuestionJson ? "Hide request JSON" : "Edit request JSON"}</button>
              </div>
              {showQuestionJson && <div className="field json-field"><label>Request JSON</label><textarea className="json-area" value={questionJson} onChange={(e) => setQuestionJson(e.target.value)} /></div>}
              <ResultBox result={results.questions} />
            </section>
            <aside className="panel span-4 info-panel"><div className="info-icon"><FileQuestion size={20} /></div><h3>From notes to questions</h3><p>Send source material to your existing API and inspect the generated JSON response here.</p><div className="endpoint-chip">POST /api/questions/generate</div><div className="tip"><b>Tip</b><span>Use real lesson content to evaluate the quality of the generated questions. Review AI output before using it in exams.</span></div></aside>
          </div>
        )}

        {activePage === "rag" && <JsonTool title="Ask Study Material" subtitle="Ask a question through the RAG endpoint." icon={BookOpen} endpoint={endpoints.rag} toolKey="rag" body={jsonBodies.rag} setBody={(v) => setJsonBodies((s) => ({ ...s, rag: v }))} run={() => runJsonTool("rag")} result={results.rag} busy={busy === "rag"} />}
        {activePage === "evaluation" && <JsonTool title="Answer Evaluation" subtitle="Evaluate a student's answer against a reference answer or rubric." icon={ClipboardCheck} endpoint={endpoints.evaluation} toolKey="evaluation" body={jsonBodies.evaluation} setBody={(v) => setJsonBodies((s) => ({ ...s, evaluation: v }))} run={() => runJsonTool("evaluation")} result={results.evaluation} busy={busy === "evaluation"} />}
        {activePage === "batch" && <JsonTool title="Batch Evaluation" subtitle="Evaluate multiple student answers in one request." icon={CheckCheck} endpoint={endpoints.batch} toolKey="batch" body={jsonBodies.batch} setBody={(v) => setJsonBodies((s) => ({ ...s, batch: v }))} run={() => runJsonTool("batch")} result={results.batch} busy={busy === "batch"} />}
        {activePage === "plagiarism" && <JsonTool title="Plagiarism Check" subtitle="Compare submitted text through your backend." icon={ScanText} endpoint={endpoints.plagiarism} toolKey="plagiarism" body={jsonBodies.plagiarism} setBody={(v) => setJsonBodies((s) => ({ ...s, plagiarism: v }))} run={() => runJsonTool("plagiarism")} result={results.plagiarism} busy={busy === "plagiarism"} />}
        <footer className="footer">ScoreMatrix · Local prototype · Human review recommended for AI-generated assessments</footer>
      </main>
    </div>
  );
}

function Dashboard({ health, refreshHealth, go }) {
  const data = health.data;
  return (
    <>
      {!data && !health.loading && <div className="alert error-alert"><CircleHelp size={17} /> {health.error} <button onClick={refreshHealth}>Retry</button></div>}
      <div className="stats-grid">
        <Metric icon={Activity} label="Backend status" value={health.loading ? "Checking…" : data ? "Online" : "Offline"} detail={data ? `API version ${data.version || "—"}` : "FastAPI health endpoint"} />
        <Metric icon={Sparkles} label="AI provider" value={data ? (data.gemini_configured ? "Configured" : "Not configured") : "—"} detail="Gemini configuration" />
        <Metric icon={ScanText} label="OCR service" value={data ? (data.ocr_available ? "Available" : "Unavailable") : "—"} detail="Scanned document support" />
      </div>
      <div className="content-grid dashboard-grid">
        <section className="panel span-7">
          <PanelHeading icon={Gauge} title="Your workspace" subtitle="Choose a tool to get started." />
          <div className="tool-cards">
            <ToolCard icon={FileQuestion} title="Generate questions" desc="Create MCQs and other questions from source text." onClick={() => go("questions")} />
            <ToolCard icon={BookOpen} title="Ask study material" desc="Send a question to the RAG assistant." onClick={() => go("rag")} />
            <ToolCard icon={ClipboardCheck} title="Evaluate answers" desc="Grade an answer using your evaluation API." onClick={() => go("evaluation")} />
            <ToolCard icon={ScanText} title="Check plagiarism" desc="Inspect text similarity results." onClick={() => go("plagiarism")} />
          </div>
        </section>
        <section className="panel span-5">
          <PanelHeading icon={Activity} title="API endpoints" subtitle="Routes available in your backend." />
          <div className="route-list">
            {["GET /health", "POST /api/questions/generate", "POST /api/rag/ask", "POST /api/evaluation/evaluate", "POST /api/evaluation/evaluate-batch", "POST /api/plagiarism/check"].map((route) => <div className="route" key={route}><span className="route-dot" /><code>{route}</code></div>)}
          </div>
          <div className="button-row"><button className="button secondary" onClick={refreshHealth}>Refresh connection</button><button className="button primary" onClick={() => window.open(`${API_BASE}/docs`, "_blank")}>Open API docs ↗</button></div>
          <p className="small-note">If a tool returns HTTP 422, update its request JSON to match the exact schema in Swagger.</p>
        </section>
      </div>
    </>
  );
}

function Metric({ icon: Icon, label, value, detail }) {
  return <section className="panel metric-card"><div className="metric-top"><span>{label}</span><div className="metric-icon"><Icon size={19} /></div></div><div className="metric-value">{value}</div><div className="metric-detail">{detail}</div></section>;
}
function PanelHeading({ icon: Icon, title, subtitle }) {
  return <div className="panel-heading"><div className="panel-icon"><Icon size={19} /></div><div><h2>{title}</h2><p>{subtitle}</p></div></div>;
}
function ToolCard({ icon: Icon, title, desc, onClick }) {
  return <button className="tool-card" onClick={onClick}><span className="tool-icon"><Icon size={19} /></span><span className="tool-copy"><b>{title}</b><small>{desc}</small></span><ChevronRight size={17} className="tool-arrow" /></button>;
}
function ResultBox({ result }) {
  if (!result) return null;
  return <div className={`result-box ${result.ok ? "result-success" : "result-error"}`}><div className="result-title">{result.ok ? "Request successful" : "Request failed"} <span>{result.ok ? "HTTP 200" : "Check response"}</span></div><pre>{JSON.stringify(result.data, null, 2)}</pre></div>;
}
function JsonTool({ title, subtitle, icon: Icon, endpoint, toolKey, body, setBody, run, result, busy }) {
  return <div className="content-grid"><section className="panel span-8"><PanelHeading icon={Icon} title={title} subtitle={subtitle} /><div className="field"><label>Request JSON — edit to match your backend schema</label><textarea className="json-area" value={body} onChange={(e) => setBody(e.target.value)} spellCheck="false" /></div><div className="button-row"><button className="button primary" onClick={run} disabled={busy}>{busy ? <LoaderCircle size={16} className="spin" /> : <Sparkles size={16} />}{busy ? "Working…" : "Run request"}</button><button className="button secondary" onClick={() => window.open(`${API_BASE}/docs`, "_blank")}>View API schema ↗</button></div><ResultBox result={result} /></section><aside className="panel span-4 info-panel"><div className="info-icon"><ShieldCheck size={20} /></div><h3>Connected endpoint</h3><p>This tool sends your JSON body directly to your local FastAPI service.</p><div className="endpoint-chip">POST {endpoint}</div><div className="tip"><b>Validation tip</b><span>Starter request fields may not match your exact Pydantic schema. If the API returns 422, use the request model displayed in Swagger and edit the JSON here.</span></div></aside></div>;
}

createRoot(document.getElementById("root")).render(<App />);
