(() => {
  "use strict";

  const HISTORY_KEY = "air-quality-lab-history-v1";
  const HISTORY_LIMIT = 20;
  const COLORS = {
    good: ["#e6f4ea", "#24734c"], satisfactory: ["#eef5df", "#63832b"],
    moderate: ["#fff4d8", "#9a6a13"], poor: ["#fff0df", "#a65a19"],
    "very poor": ["#fce8e7", "#a93e37"], severe: ["#f5e6e8", "#842e38"],
    low: ["#e6f4ea", "#24734c"], medium: ["#fff4d8", "#9a6a13"],
    high: ["#fce8e7", "#a93e37"]
  };
  const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  const $ = (id) => document.getElementById(id);
  let metrics = null;
  let history = [];
  let latestPrediction = null;

  const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
  const number = (value, digits = 1) => Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits, maximumFractionDigits: digits
  });
  const categoryStyle = (category) => COLORS[String(category).toLowerCase()] || ["#edf2ef", "#50665c"];

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => {
        const active = item === tab;
        item.classList.toggle("active", active);
        item.setAttribute("aria-selected", String(active));
      });
      document.querySelectorAll(".panel").forEach((panel) => {
        const active = panel.id === tab.dataset.tab;
        panel.classList.toggle("active", active);
        panel.hidden = !active;
      });
    });
  });

  if (!$("form")) return;

  fetch("/api/metrics")
    .then((response) => {
      if (!response.ok) throw new Error("Model information is not available. Check that training has completed.");
      return response.json();
    })
    .then((data) => {
      metrics = data;
      buildForm();
      fillInsights();
      setValues("median");
      loadHistory();
      renderHistory();
    })
    .catch((error) => showError(error.message || "Could not load model information."));

  function showError(message) {
    const error = $("error");
    if (!error) return;
    error.textContent = message;
    error.hidden = !message;
  }

  function buildField(name, kind, options = {}) {
    const box = $("fields");
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const label = document.createElement("label");
    const id = `feature-${box.children.length}`;
    label.htmlFor = id;
    label.append(document.createTextNode(name));
    if (options.hint) {
      const hint = document.createElement("small");
      hint.textContent = options.hint;
      label.append(hint);
    }

    let input;
    if (kind === "categorical" || kind === "month") {
      input = document.createElement("select");
      const choices = kind === "month"
        ? MONTHS.map((month, index) => [String(index + 1), month])
        : options.choices.map((choice) => [String(choice), String(choice)]);
      choices.forEach(([value, text]) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = text;
        input.append(option);
      });
      if (options.default !== undefined) input.value = String(options.default);
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.inputMode = "decimal";
      input.placeholder = `Typical: ${options.median}`;
      if (Number.isFinite(Number(options.min))) input.min = String(options.min);
      if (Number.isFinite(Number(options.max))) input.max = String(options.max);
      input.title = `Training range: ${options.min} to ${options.max}; median ${options.median}`;
    }
    input.id = id;
    input.dataset.name = name;
    input.dataset.kind = kind;
    wrapper.append(label, input);
    box.append(wrapper);
  }

  function buildForm() {
    const schema = metrics.schema;
    $("fields").textContent = "";
    schema.categorical.forEach((field) => buildField(field.name, "categorical", field));
    if (schema.has_month) buildField("Month", "month", { default: new Date().getMonth() + 1 });
    schema.numeric.forEach((field) => {
      buildField(field.name, "numeric", { ...field, hint: `median ${number(field.median, 2)}` });
    });
    $("target-label").textContent = `Target · ${metrics.dataset.target}`;
  }

  function setValues(mode) {
    metrics.schema.numeric.forEach((field) => {
      const input = [...document.querySelectorAll("#fields [data-name]")].find((item) => item.dataset.name === field.name);
      if (input) input.value = mode === "median" ? field.median : field.p95;
    });
    const month = [...document.querySelectorAll("#fields [data-name]")].find((item) => item.dataset.name === "Month");
    if (month) month.value = mode === "median" ? String(new Date().getMonth() + 1) : "12";
  }

  $("btn-typical").addEventListener("click", () => setValues("median"));
  $("btn-polluted").addEventListener("click", () => setValues("p95"));

  $("form").addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("");
    const features = {};
    document.querySelectorAll("#fields [data-name]").forEach((input) => {
      features[input.dataset.name] = input.value;
    });
    const button = $("btn-predict");
    button.disabled = true;
    button.querySelector("span").textContent = "Estimating…";
    $("result").setAttribute("aria-busy", "true");
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Prediction failed.");
      latestPrediction = { ...data, features };
      renderResult(data);
      if (saveHistory(latestPrediction)) renderHistory();
    } catch (error) {
      showError(error.message || "Prediction failed. Please try again.");
    } finally {
      button.disabled = false;
      button.querySelector("span").textContent = "Run prediction";
      $("result").setAttribute("aria-busy", "false");
    }
  });

  function renderResult(data) {
    const [background, foreground] = categoryStyle(data.category);
    const confidence = Math.max(0, Math.min(100, Number(data.confidence) || 0));
    const max = Number(metrics.dataset.target_max) || 500;
    const marker = Math.max(0, Math.min(100, Number(data.value) / max * 100));
    const probabilities = (data.probabilities || []).map((item) =>
      `<li><span>${escapeHTML(item.label)}</span><span class="prob-track"><span class="prob-fill" style="width:${Math.max(0, Math.min(100, Number(item.p) || 0))}%"></span></span><span>${number(item.p)}%</span></li>`
    ).join("");

    $("result").innerHTML = `
      <div class="result-topline"><span class="eyebrow">Your estimate</span><span class="result-icon" aria-hidden="true">↗</span></div>
      <div class="big">${number(data.value, 1)}</div>
      <p class="meta">Estimated ${escapeHTML(data.target)}</p>
      <span class="cat" style="background:${background};color:${foreground}"><i class="cat-dot" aria-hidden="true"></i>${escapeHTML(data.category)}</span>
      <div class="scale" aria-label="Estimate position on the target range">
        <div class="scale-track"><i class="scale-marker" style="left:${marker}%"></i></div>
        <div class="scale-labels"><span>0</span><span>${number(max / 2, 0)}</span><span>${number(max, 0)}</span></div>
      </div>
      <div class="confidence-row"><span>Category confidence</span><b>${number(confidence)}%</b></div>
      <div class="confidence-track"><span class="confidence-fill" style="width:${confidence}%"></span></div>
      ${data.advice ? `<p class="advice">${escapeHTML(data.advice)}</p>` : ""}
      ${probabilities ? `<ul class="probs" aria-label="Category probability estimates">${probabilities}</ul>` : ""}
      <div class="result-footer"><small>Saved to your local prediction history</small><button type="button" class="copy-button" id="btn-copy-result">Copy result</button></div>`;
  }

  $("result").addEventListener("click", async (event) => {
    if (event.target.id !== "btn-copy-result" || !latestPrediction) return;
    const text = `${latestPrediction.target}: ${number(latestPrediction.value, 1)} | ${latestPrediction.category} (${number(latestPrediction.confidence)}% confidence)`;
    try {
      await navigator.clipboard.writeText(text);
      event.target.textContent = "Copied";
    } catch {
      showError("Could not copy the result. Your browser may block clipboard access.");
    }
  });

  function loadHistory() {
    try {
      const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      history = Array.isArray(saved) ? saved.filter((item) =>
        item && typeof item.timestamp === "string" && Number.isFinite(Number(item.value))
      ).slice(0, HISTORY_LIMIT) : [];
    } catch {
      history = [];
      showError("Saved history could not be read. New predictions can still be made.");
    }
  }

  function saveHistory(prediction) {
    history.unshift({
      timestamp: new Date().toISOString(),
      target: prediction.target,
      value: Number(prediction.value),
      category: String(prediction.category),
      confidence: Number(prediction.confidence),
      features: prediction.features
    });
    history = history.slice(0, HISTORY_LIMIT);
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
      return true;
    } catch {
      history.shift();
      showError("Prediction worked, but this browser could not save its history.");
      return false;
    }
  }

  function renderHistory() {
    $("history-count").textContent = String(history.length);
    $("history-empty").hidden = history.length > 0;
    $("history-list").innerHTML = history.map((item) => {
      const [background, foreground] = categoryStyle(item.category);
      const date = new Date(item.timestamp);
      return `<article class="history-item">
        <div><span class="history-time">${escapeHTML(date.toLocaleString())}</span></div>
        <div class="history-value">${number(item.value, 1)}<small>${escapeHTML(item.target)}</small></div>
        <div class="history-category"><span class="cat" style="background:${background};color:${foreground}">${escapeHTML(item.category)}</span><span class="meta">${number(item.confidence)}% confidence</span></div>
      </article>`;
    }).join("");
    if (history.length) {
      const average = history.reduce((total, item) => total + Number(item.value), 0) / history.length;
      const highest = history.reduce((best, item) => Number(item.value) > Number(best.value) ? item : best, history[0]);
      $("history-summary").innerHTML = [
        stat(history.length, "saved predictions"),
        stat(number(average, 1), "average estimate", metrics.dataset.target),
        stat(number(highest.value, 1), "highest estimate", highest.category)
      ].join("");
    } else {
      $("history-summary").textContent = "";
    }
  }

  function stat(value, label, detail = "") {
    return `<div class="stat"><span class="stat-value">${escapeHTML(value)}</span><span class="stat-label">${escapeHTML(label)}</span>${detail ? `<span class="stat-detail">${escapeHTML(detail)}</span>` : ""}</div>`;
  }

  $("btn-clear-history").addEventListener("click", () => {
    if (!history.length || !confirm("Clear all saved predictions from this browser?")) return;
    try {
      localStorage.removeItem(HISTORY_KEY);
      history = [];
      renderHistory();
      showError("");
    } catch {
      showError("Could not clear saved history in this browser.");
    }
  });

  $("btn-export-history").addEventListener("click", () => {
    if (!history.length) {
      showError("There are no saved predictions to export yet.");
      return;
    }
    const featureNames = [...new Set(history.flatMap((item) => Object.keys(item.features || {})))];
    const columns = ["timestamp", "target", "value", "category", "confidence", ...featureNames];
    const csvCell = (value) => {
      let text = String(value ?? "");
      if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`;
      return `"${text.replace(/"/g, '""')}"`;
    };
    const rows = [columns, ...history.map((item) => [
      item.timestamp, item.target, item.value, item.category, item.confidence,
      ...featureNames.map((name) => item.features?.[name] ?? "")
    ])];
    downloadFile("air-quality-predictions.csv", rows.map((row) => row.map(csvCell).join(",")).join("\r\n"), "text/csv;charset=utf-8");
  });

  $("btn-download-report").addEventListener("click", () => {
    downloadFile("air-quality-model-report.json", JSON.stringify(metrics, null, 2), "application/json;charset=utf-8");
  });

  function downloadFile(filename, content, type) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function fillInsights() {
    const data = metrics.dataset;
    const regression = metrics.regression;
    const classification = metrics.classification;
    const bestReg = regression.results.find((result) => result.model === regression.best);
    const bestCls = classification.results.find((result) => result.model === classification.best);
    $("summary").innerHTML = [
      stat(Number(data.rows).toLocaleString(), "training records", data.file),
      stat(data.features.length, "model input features", `${data.columns} original columns`),
      stat(number(bestReg.r2, 3), "best regression R²", regression.best),
      stat(`${number(bestCls.accuracy * 100)}%`, "category accuracy", classification.best),
      stat(number(data.target_mean), "average target", data.target),
      stat(`${number(data.missing_pct)}%`, "missing input values", "Imputed during training")
    ].join("");

    $("reg-table").innerHTML = `<thead><tr><th>Model</th><th class="num">R²</th><th class="num">RMSE</th><th class="num">MAE</th></tr></thead><tbody>` +
      regression.results.map((result) => `<tr class="${result.model === regression.best ? "best" : ""}"><td>${escapeHTML(result.model)}</td><td class="num">${number(result.r2, 3)}</td><td class="num">${number(result.rmse, 2)}</td><td class="num">${number(result.mae, 2)}</td></tr>`).join("") + "</tbody>";
    $("cls-table").innerHTML = `<thead><tr><th>Model</th><th class="num">Accuracy</th><th class="num">Weighted F1</th></tr></thead><tbody>` +
      classification.results.map((result) => `<tr class="${result.model === classification.best ? "best" : ""}"><td>${escapeHTML(result.model)}</td><td class="num">${number(result.accuracy * 100)}%</td><td class="num">${number(result.f1, 3)}</td></tr>`).join("") + "</tbody>";

    const hiddenCharts = new Set([
      "eda_target_distribution.png",
      "eda_correlation.png",
      "eda_monthly.png",
      "eda_by_category.png"
    ]);
    $("charts").innerHTML = metrics.charts.filter((chart) => !hiddenCharts.has(chart.file)).map((chart) =>
      `<figure class="chart-card"><img src="/static/charts/${encodeURIComponent(chart.file)}" alt="${escapeHTML(chart.title)}" loading="lazy"><figcaption><b>${escapeHTML(chart.title)}</b>${escapeHTML(chart.caption)}</figcaption></figure>`
    ).join("");
  }
})();
