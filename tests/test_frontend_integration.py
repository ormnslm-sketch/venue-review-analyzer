from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"


def read(relative_path: str) -> str:
    return (FRONTEND / relative_path).read_text(encoding="utf-8")


class FrontendIntegrationTests(unittest.TestCase):
    def test_legacy_places_ui_is_preserved(self) -> None:
        index = read("index.html")
        app = read("js/app.js")
        details = read("js/details.js")

        self.assertIn('id="search-input"', index)
        self.assertIn('id="category-filter"', index)
        self.assertIn('id="district-filter"', index)
        self.assertIn('fetch("http://127.0.0.1:8000/places")', app)
        self.assertIn("http://127.0.0.1:8000/places/${selectedId}", details)

    def test_review_analyzer_uses_final_api_contract(self) -> None:
        index = read("index.html")
        analyzer = read("js/analyzer.js")

        self.assertIn('id="review-text"', index)
        self.assertIn('id="analyze-review-button"', index)
        self.assertIn('id="analysis-results"', index)
        self.assertIn('http://127.0.0.1:8000/analyze-review', analyzer)
        self.assertIn('method: "POST"', analyzer)
        self.assertIn("JSON.stringify({text})", analyzer)
        self.assertIn('["taste", "atmosphere", "service", "value"]', analyzer)

    def test_review_analyzer_handles_loading_and_errors(self) -> None:
        analyzer = read("js/analyzer.js")
        i18n = read("js/i18n.js")

        self.assertIn('t("analyzer.loading")', analyzer)
        self.assertIn('showAnalysisError("analyzer.error.empty")', analyzer)
        self.assertIn('"analyzer.error.network"', analyzer)
        self.assertIn('"analyzer.error.invalidJson"', analyzer)
        self.assertIn("Analyzing...", i18n)
        self.assertIn("Analiz ediliyor...", i18n)

    def test_english_and_turkish_i18n_contract(self) -> None:
        index = read("index.html")
        details = read("details.html")
        i18n = read("js/i18n.js")

        self.assertIn('<html lang="en">', index)
        self.assertIn('data-ui-language="en"', index)
        self.assertIn('data-ui-language="tr"', index)
        self.assertLess(index.index("js/i18n.js"), index.index("js/app.js"))
        self.assertLess(details.index("js/i18n.js"), details.index("js/details.js"))
        self.assertIn('const DEFAULT_LANGUAGE = "en"', i18n)
        self.assertIn('const STORAGE_KEY = "uiLanguage"', i18n)
        self.assertIn('"aspect.taste": "Taste"', i18n)
        self.assertIn('"aspect.taste": "Lezzet"', i18n)
        self.assertIn('"analyzer.notMentioned": "Not mentioned"', i18n)
        self.assertIn('"analyzer.notMentioned": "Bahsedilmedi"', i18n)

    def test_backend_contract_values_remain_untranslated(self) -> None:
        analyzer = read("js/analyzer.js")
        app = read("js/app.js")

        self.assertIn('["taste", "atmosphere", "service", "value"]', analyzer)
        self.assertIn('`sentiment.${result.sentiment}`', analyzer)
        self.assertIn('selectedCategory === "all"', app)
        self.assertIn('selectedDistrict === "all"', app)


if __name__ == "__main__":
    unittest.main()
