const ANALYZE_REVIEW_URL = "http://127.0.0.1:8000/analyze-review";

const reviewText = document.querySelector("#review-text");
const analyzeButton = document.querySelector("#analyze-review-button");
const analysisStatus = document.querySelector("#analysis-status");
const analysisError = document.querySelector("#analysis-error");
const analysisResults = document.querySelector("#analysis-results");
const aspectResults = document.querySelector("#aspect-results");
const analysisDetailsContent = document.querySelector(
  "#analysis-details-content"
);
let latestAnalysis = null;
let analysisState = "idle";
let analysisErrorKey = null;

function t(key, values) {
  return window.I18n.t(key, values);
}

function showAnalysisError(key) {
  analysisErrorKey = key;
  analysisState = "error";
  analysisError.textContent = t(key);
  analysisError.hidden = false;
  analysisResults.hidden = true;
}

function clearAnalysisError() {
  analysisErrorKey = null;
  analysisError.textContent = "";
  analysisError.hidden = true;
}

function renderAspectResult(aspect, result) {
  const card = document.createElement("article");
  card.className = "aspect-result-card";

  const title = document.createElement("h3");
  title.textContent = t(`aspect.${aspect}`);
  card.appendChild(title);

  const outcome = document.createElement("p");
  if (!result.present) {
    outcome.className = "aspect-outcome sentiment-absent";
    outcome.textContent = t("analyzer.notMentioned");
    card.appendChild(outcome);
    return card;
  }

  outcome.className = `aspect-outcome sentiment-${result.sentiment}`;
  outcome.textContent = t(`sentiment.${result.sentiment}`);
  card.appendChild(outcome);

  const confidence = document.createElement("p");
  confidence.className = "aspect-confidence";
  confidence.textContent = t("analyzer.confidence", {
    value: Math.round(result.sentiment_confidence * 100)
  });
  card.appendChild(confidence);
  return card;
}

function renderAnalysisDetails(data) {
  analysisDetailsContent.innerHTML = "";

  const summary = document.createElement("p");
  summary.textContent = t("analyzer.detailsSummary", {
    sentences: data.sentence_count,
    calls: data.sentiment_calls
  });
  analysisDetailsContent.appendChild(summary);

  const sentenceList = document.createElement("ol");
  data.sentences.forEach(function (sentence) {
    const item = document.createElement("li");
    item.textContent = sentence;
    sentenceList.appendChild(item);
  });
  analysisDetailsContent.appendChild(sentenceList);

  Object.entries(data.aspects).forEach(function ([aspect, result]) {
    if (!result.present || result.evidence_sentences.length === 0) {
      return;
    }
    const heading = document.createElement("h4");
    heading.textContent = t("analyzer.evidence", {
      aspect: t(`aspect.${aspect}`)
    });
    analysisDetailsContent.appendChild(heading);

    const evidenceList = document.createElement("ul");
    result.evidence_sentences.forEach(function (evidence) {
      const item = document.createElement("li");
      item.textContent = evidence.text;
      evidenceList.appendChild(item);
    });
    analysisDetailsContent.appendChild(evidenceList);
  });
}

function renderAnalysis(data) {
  latestAnalysis = data;
  aspectResults.innerHTML = "";
  ["taste", "atmosphere", "service", "value"].forEach(function (aspect) {
    aspectResults.appendChild(renderAspectResult(aspect, data.aspects[aspect]));
  });
  renderAnalysisDetails(data);
  analysisResults.hidden = false;
}

function refreshAnalyzerLanguage() {
  if (latestAnalysis && !analysisResults.hidden) {
    renderAnalysis(latestAnalysis);
  }
  if (analysisState === "loading") {
    analysisStatus.textContent = t("analyzer.loading");
  } else if (analysisState === "complete") {
    analysisStatus.textContent = t("analyzer.complete");
  } else if (analysisState === "error" && analysisErrorKey) {
    analysisError.textContent = t(analysisErrorKey);
  }
}

async function analyzeReview() {
  const text = reviewText.value.trim();
  clearAnalysisError();

  if (!text) {
    showAnalysisError("analyzer.error.empty");
    reviewText.focus();
    return;
  }

  analyzeButton.disabled = true;
  analysisState = "loading";
  analysisStatus.textContent = t("analyzer.loading");
  analysisResults.hidden = true;

  try {
    const response = await fetch(ANALYZE_REVIEW_URL, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text})
    });

    let payload;
    try {
      payload = await response.json();
    } catch (error) {
      throw new Error("analyzer.error.invalidJson");
    }

    if (!response.ok) {
      throw new Error("analyzer.error.api");
    }

    if (!payload.aspects) {
      throw new Error("analyzer.error.missingAspects");
    }

    renderAnalysis(payload);
    analysisState = "complete";
    analysisStatus.textContent = t("analyzer.complete");
  } catch (error) {
    const knownKeys = new Set([
      "analyzer.error.invalidJson",
      "analyzer.error.api",
      "analyzer.error.missingAspects"
    ]);
    const key = error instanceof TypeError || error.message === "Failed to fetch"
      ? "analyzer.error.network"
      : knownKeys.has(error.message)
        ? error.message
        : "analyzer.error.unexpected";
    showAnalysisError(key);
    analysisStatus.textContent = "";
  } finally {
    analyzeButton.disabled = false;
  }
}

analyzeButton.addEventListener("click", analyzeReview);
window.addEventListener("ui-language-changed", refreshAnalyzerLanguage);
