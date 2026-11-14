import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, ArrowRight, BrainCircuit, Check, ChevronDown, FileImage, GitCompare, ImagePlus, Info, Menu, RefreshCw, Sparkles, Upload, X } from "lucide-react";
import { motion } from "framer-motion";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const steps = [
  ["01", "Image upload", "Select two supported images to compare.", Upload],
  ["02", "Preprocessing", "Images become consistent RGB tensors.", FileImage],
  ["03", "Feature extraction", "MobileNetV2 reads visual features.", BrainCircuit],
  ["04", "Cosine similarity", "Embeddings are compared mathematically.", GitCompare],
  ["05", "Results", "Explore a clear, explainable score.", Activity],
];

function Logo() { return <a className="logo" href="#home"><span className="logo-mark"><Sparkles size={18}/></span><span>SYNAPSE</span></a>; }

function Navbar() {
  const [open, setOpen] = useState(false);
  return <nav><Logo/><button className="menu" onClick={() => setOpen(!open)} aria-label="Toggle menu"><Menu/></button><div className={`nav-links ${open ? "open" : ""}`}><a href="#home">Home</a><a href="#workspace">Compare Images</a><a href="#how-it-works">How It Works</a><a href="#about">About</a></div><a className="nav-cta" href="#workspace">Start comparing <ArrowRight size={16}/></a></nav>;
}

function UploadCard({ label, item, onFile, onRemove }) {
  const inputId = `image-${label.toLowerCase()}`;
  const choose = (event) => onFile(event.target.files?.[0]);
  const drop = (event) => { event.preventDefault(); onFile(event.dataTransfer.files?.[0]); };
  return <div className={`upload-card ${item ? "has-file" : ""}`} onDragOver={(e) => e.preventDefault()} onDrop={drop}>
    <div className="card-heading"><span>{label}</span>{item && <button className="icon-btn" onClick={onRemove} aria-label={`Remove ${label}`}><X size={16}/></button>}</div>
    {item ? <div className="preview"><img src={item.url} alt={`${label} preview`}/><div className="file-meta"><strong>{item.file.name}</strong><span>{(item.file.size / 1024 / 1024).toFixed(2)} MB · {item.width} × {item.height}px</span></div></div> :
      <label className="drop-zone" htmlFor={inputId}><span className="upload-icon"><ImagePlus/></span><strong>Drop your image here</strong><span>or <u>browse files</u></span><small>JPG, PNG, WEBP · max 10 MB</small><input id={inputId} type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" onChange={choose}/></label>}
  </div>;
}

function Score({ result }) {
  const index = result.normalized_similarity_index;
  return <motion.div className="result-card" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }}>
    <div className="score-ring" style={{ "--score": `${index * 3.6}deg` }}><div><strong>{result.cosine_similarity.toFixed(4)}</strong><span>raw cosine</span></div></div>
    <div className="score-copy"><span className="eyebrow">ANALYSIS COMPLETE</span><h3>{result.classification}</h3><p>{result.explanation}</p><div className="metrics"><span><b>Normalized index</b>{index.toFixed(1)} / 100</span><span><b>Model</b>{result.model_name}</span><span><b>Algorithm</b>{result.algorithm_name}</span></div><small className="score-note">{result.score_interpretation}</small></div>
  </motion.div>;
}

function App() {
  const [images, setImages] = useState([null, null]); const [loading, setLoading] = useState(false); const [result, setResult] = useState(null); const [error, setError] = useState("");
  const setFile = (index, file) => {
    setError(""); if (!file) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type) || file.size > 10 * 1024 * 1024) { setError("Please choose a JPG, PNG, or WEBP image under 10 MB."); return; }
    const url = URL.createObjectURL(file); const image = new Image(); image.onload = () => { if (image.width < 32 || image.height < 32) { setError("Images must be at least 32 × 32 pixels."); return; } setImages((old) => { const next = [...old]; next[index] = { file, url, width: image.width, height: image.height }; return next; }); }; image.onerror = () => setError("That file could not be decoded as an image."); image.src = url;
  };
  const compare = async () => {
    if (!images[0] || !images[1]) { setError("Upload both images before starting an analysis."); return; }
    setLoading(true); setError(""); setResult(null); const body = new FormData(); body.append("image1", images[0].file); body.append("image2", images[1].file);
    try { const response = await fetch(`${API_URL}/api/compare`, { method: "POST", body }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || "The analysis could not be completed."); setResult(data); setTimeout(() => document.getElementById("results")?.scrollIntoView({ behavior: "smooth" }), 50); } catch (e) { setError(e.message); } finally { setLoading(false); }
  };
  const reset = () => { images.forEach((item) => item && URL.revokeObjectURL(item.url)); setImages([null, null]); setResult(null); setError(""); };
  return <><Navbar/><main id="home">
    <section className="hero"><div className="hero-copy"><span className="eyebrow"><span className="pulse"></span> DEEP LEARNING · VISUAL ANALYSIS</span><h1>Two images.<br/><em>One intelligence.</em></h1><p>Discover the hidden similarities between images using deep learning. Upload, compare, and explore visual connections powered by AI.</p><div className="hero-actions"><a className="button primary" href="#workspace">Start comparing <ArrowRight size={18}/></a><a className="button secondary" href="#how-it-works">Explore how it works <ChevronDown size={17}/></a></div></div><div className="hero-art"><div className="art-image art-a"><div className="art-lines"></div><span>IMAGE A</span></div><div className="art-node"><span></span><span></span><span></span><BrainCircuit/></div><div className="art-image art-b"><div className="art-lines"></div><span>IMAGE B</span></div><div className="art-caption"><Check size={14}/> AI-powered visual intelligence</div></div></section>
    <section className="workspace section" id="workspace"><div className="section-title"><span className="eyebrow">01 / WORKSPACE</span><h2>Visual intelligence <em>workspace.</em></h2><p>Upload two images and let our AI analyse their visual similarity.</p></div><div className="upload-grid"><UploadCard label="IMAGE A" item={images[0]} onFile={(f) => setFile(0, f)} onRemove={() => setImages([null, images[1]])}/><div className="versus">VS</div><UploadCard label="IMAGE B" item={images[1]} onFile={(f) => setFile(1, f)} onRemove={() => setImages([images[0], null])}/></div>{error && <div className="error"><Info size={16}/>{error}</div>}<div className="action-row"><button className="button primary analyse" onClick={compare} disabled={loading}>{loading ? <><RefreshCw className="spin" size={18}/> Analysing...</> : <>Analyse similarity <Sparkles size={17}/></>}</button>{(result || images.some(Boolean)) && <button className="text-button" onClick={reset}>Reset workspace</button>}</div>{loading && <div className="loading"><div className="loading-orbit"><BrainCircuit/></div><strong>Analysing visual features...</strong><span>Extracting deep learning embeddings and calculating similarity.</span></div>}</section>
    {result && <section className="results section" id="results"><div className="section-title"><span className="eyebrow">02 / RESULTS</span><h2>A connection, <em>revealed.</em></h2><p>Here is what the model found in your image pair.</p></div><Score result={result}/><div className="result-images">{images.map((item, i) => <div key={i}><span>IMAGE {i ? "B" : "A"}</span><img src={item.url} alt={`Comparison ${i ? "B" : "A"}`}/></div>)}</div><button className="button secondary center" onClick={reset}>Compare another pair <ArrowRight size={17}/></button></section>}
    <section className="how section" id="how-it-works"><div className="section-title"><span className="eyebrow">03 / THE PROCESS</span><h2>From pixels to <em>perspective.</em></h2><p>A transparent five-step pipeline turns image data into an understandable result.</p></div><div className="steps">{steps.map(([number, title, copy, Icon]) => <div className="step" key={number}><span className="step-number">{number}</span><Icon size={22}/><h3>{title}</h3><p>{copy}</p></div>)}</div></section>
    <section className="about section" id="about"><div className="about-card"><div><span className="eyebrow">04 / ABOUT SYNAPSE</span><h2>Similarity is more than <em>looking alike.</em></h2></div><p>SYNAPSE uses MobileNetV2, a convolutional neural network trained on ImageNet, to turn each image into a feature embedding. Cosine similarity compares those embeddings, capturing visual concepts beyond raw pixels. It can support duplicate detection, image retrieval, asset organization and computer vision research.</p></div></section>
  </main><footer><Logo/><p>Where images connect. Intelligence reveals.</p><span>Built as an academic computer vision project · {new Date().getFullYear()}</span></footer></>;
}

createRoot(document.getElementById("root")).render(<App />);
