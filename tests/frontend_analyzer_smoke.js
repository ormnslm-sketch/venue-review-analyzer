const fs = require("fs");
const vm = require("vm");

class Element {
  constructor(dataset = {}) {
    this.children = [];
    this.className = "";
    this.dataset = dataset;
    this.disabled = false;
    this.hidden = false;
    this.listeners = {};
    this.placeholder = "";
    this.textContent = "";
    this.value = "";
    this.attributes = {};
    this.classList = {
      toggle: (name, enabled) => {
        this.attributes[`class:${name}`] = enabled;
      }
    };
    this._innerHTML = "";
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set innerHTML(value) {
    this._innerHTML = value;
    if (value === "") {
      this.children = [];
    }
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  addEventListener(event, callback) {
    this.listeners[event] = callback;
  }

  setAttribute(name, value) {
    this.attributes[name] = value;
  }

  focus() {}
}

function makePayload(text) {
  return {
    text,
    sentence_count: 1,
    sentences: [text],
    sentence_aspect_probabilities: [],
    aspects: {
      taste: {
        present: true,
        sentiment: "positive",
        sentiment_confidence: 0.91,
        evidence_sentences: [{text}]
      },
      atmosphere: {present: false, evidence_sentences: []},
      service: {
        present: true,
        sentiment: "negative",
        sentiment_confidence: 0.82,
        evidence_sentences: [{text}]
      },
      value: {present: false, evidence_sentences: []}
    },
    no_target_aspects_detected: false,
    sentiment_calls: 2
  };
}

async function main() {
  const selectors = [
    "#review-text",
    "#analyze-review-button",
    "#analysis-status",
    "#analysis-error",
    "#analysis-results",
    "#aspect-results",
    "#analysis-details-content"
  ];
  const elements = new Map(selectors.map((selector) => [selector, new Element()]));
  elements.get("#analyze-review-button").dataset.i18n = "analyzer.button";

  const englishButton = new Element({uiLanguage: "en"});
  const turkishButton = new Element({uiLanguage: "tr"});
  const languageButtons = [englishButton, turkishButton];
  const eventListeners = {};
  const storage = new Map();
  const requestedTexts = [];

  const document = {
    body: new Element({titleKey: "meta.indexTitle"}),
    documentElement: {lang: ""},
    readyState: "complete",
    title: "",
    addEventListener() {},
    createElement: () => new Element(),
    querySelector: (selector) => elements.get(selector),
    querySelectorAll: (selector) => {
      if (selector === "[data-ui-language]") {
        return languageButtons;
      }
      if (selector === "[data-i18n]") {
        return [...elements.values()].filter((element) => element.dataset.i18n);
      }
      return [];
    }
  };

  const context = {
    console,
    CustomEvent: class {
      constructor(type, init) {
        this.type = type;
        this.detail = init.detail;
      }
    },
    document,
    Error,
    TypeError,
    fetch: async (url, options) => {
      if (url !== "http://127.0.0.1:8000/analyze-review") {
        throw new Error(`Unexpected URL: ${url}`);
      }
      if (options.method !== "POST") {
        throw new Error(`Unexpected method: ${options.method}`);
      }
      const body = JSON.parse(options.body);
      requestedTexts.push(body.text);
      return {ok: true, json: async () => makePayload(body.text)};
    },
    localStorage: {
      getItem: (key) => storage.get(key) || null,
      setItem: (key, value) => storage.set(key, value)
    },
    addEventListener: (name, callback) => {
      eventListeners[name] = eventListeners[name] || [];
      eventListeners[name].push(callback);
    },
    dispatchEvent: (event) => {
      (eventListeners[event.type] || []).forEach((callback) => callback(event));
    }
  };
  context.window = context;

  vm.createContext(context);
  vm.runInContext(fs.readFileSync("frontend/js/i18n.js", "utf8"), context);
  vm.runInContext(fs.readFileSync("frontend/js/analyzer.js", "utf8"), context);

  if (context.I18n.getLanguage() !== "en") {
    throw new Error("First visit did not default to English.");
  }
  if (elements.get("#analyze-review-button").textContent !== "Analyze Review") {
    throw new Error("English analyzer button was not rendered.");
  }

  const turkishReview = "Kahve çok iyiydi ama servis yavaştı.";
  elements.get("#review-text").value = turkishReview;
  await elements.get("#analyze-review-button").listeners.click();
  let cards = elements.get("#aspect-results").children;
  if (cards.length !== 4 || cards[0].children[0].textContent !== "Taste") {
    throw new Error("Turkish review did not render in the English interface.");
  }

  context.I18n.setLanguage("tr");
  cards = elements.get("#aspect-results").children;
  if (storage.get("uiLanguage") !== "tr") {
    throw new Error("Turkish preference was not persisted.");
  }
  if (elements.get("#analyze-review-button").textContent !== "Yorumu Analiz Et") {
    throw new Error("Turkish analyzer button was not rendered.");
  }
  if (cards[0].children[0].textContent !== "Lezzet") {
    throw new Error("Existing analysis did not switch to Turkish.");
  }
  vm.runInContext(fs.readFileSync("frontend/js/i18n.js", "utf8"), context);
  if (context.I18n.getLanguage() !== "tr") {
    throw new Error("Stored Turkish preference was not restored after reload.");
  }

  const englishReview = "The coffee was excellent but the service was slow.";
  elements.get("#review-text").value = englishReview;
  await elements.get("#analyze-review-button").listeners.click();
  if (elements.get("#analysis-status").textContent !== "Analiz tamamlandı.") {
    throw new Error("English review did not complete in the Turkish interface.");
  }

  context.I18n.setLanguage("en");
  if (storage.get("uiLanguage") !== "en") {
    throw new Error("English preference was not persisted.");
  }
  if (requestedTexts[0] !== turkishReview || requestedTexts[1] !== englishReview) {
    throw new Error("Review language was altered by the UI language setting.");
  }

  console.log("frontend EN/TR analyzer smoke: PASS");
  console.log("default English: PASS");
  console.log("language persistence: PASS");
  console.log("TR review with EN interface: PASS");
  console.log("EN review with TR interface: PASS");
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
