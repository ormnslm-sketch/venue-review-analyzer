(function () {
  "use strict";

  const STORAGE_KEY = "uiLanguage";
  const DEFAULT_LANGUAGE = "en";
  const SUPPORTED_LANGUAGES = ["en", "tr"];

  const translations = {
    en: {
      "meta.indexTitle": "Why Go?",
      "meta.detailsTitle": "Venue Details | Why Go?",
      "site.name": "Why Go?",
      "site.tagline": "Discover Kadıköy's venues through scores.",
      "site.detailsTagline": "Venue details",
      "language.label": "Interface language",
      "page.eyebrow": "Venue guide",
      "page.heading": "Venues worth discovering",
      "analyzer.eyebrow": "Review analysis",
      "analyzer.title": "Analyze a restaurant review",
      "analyzer.description": "Explore opinions about taste, atmosphere, service and value.",
      "analyzer.inputLabel": "Restaurant review",
      "analyzer.placeholder": "Paste a restaurant review",
      "analyzer.button": "Analyze Review",
      "analyzer.loading": "Analyzing...",
      "analyzer.complete": "Analysis complete.",
      "analyzer.details": "Analysis details",
      "analyzer.detailsSummary": "{sentences} sentences analyzed; {calls} sentiment inferences made.",
      "analyzer.evidence": "{aspect} evidence",
      "analyzer.confidence": "Confidence: {value}%",
      "analyzer.notMentioned": "Not mentioned",
      "analyzer.error.empty": "Please enter a restaurant review to analyze.",
      "analyzer.error.network": "The backend could not be reached. Check that the FastAPI service is running.",
      "analyzer.error.invalidJson": "The backend did not return a valid JSON response.",
      "analyzer.error.api": "The review could not be analyzed.",
      "analyzer.error.missingAspects": "The response does not contain aspect results.",
      "analyzer.error.unexpected": "An unexpected error occurred during review analysis.",
      "aspect.taste": "Taste",
      "aspect.atmosphere": "Atmosphere",
      "aspect.service": "Service",
      "aspect.value": "Value",
      "aspect.staff": "Staff",
      "aspect.decoration": "Decoration",
      "sentiment.positive": "Positive",
      "sentiment.neutral": "Neutral",
      "sentiment.negative": "Negative",
      "filters.search": "Search venues",
      "filters.searchPlaceholder": "Search venues...",
      "filters.category": "Category",
      "filters.allCategories": "All",
      "filters.district": "District",
      "filters.allDistricts": "All districts",
      "category.cafe": "Cafe",
      "category.bakeryCafe": "Bakery Cafe",
      "category.restaurant": "Restaurant",
      "category.pub": "Pub",
      "places.category": "Category: {value}",
      "places.overallScore": "Overall score: {value}",
      "places.viewDetails": "View Details",
      "places.loadError": "The venue list is currently unavailable. Review analysis remains available.",
      "details.back": "← Back to all venues",
      "details.notFound": "Venue not found.",
      "details.loadError": "An error occurred while loading venue information.",
      "details.scores": "Scores",
      "details.priceLevel": "Price level: {value}",
      "details.coffeeBeans": "Coffee beans: {value}",
      "details.noCoffeeBeans": "Coffee bean information is unavailable.",
      "details.ourView": "Our view: {value}"
    },
    tr: {
      "meta.indexTitle": "Neden Gidelim?",
      "meta.detailsTitle": "Mekan Detayları | Neden Gidelim?",
      "site.name": "Neden Gidelim?",
      "site.tagline": "Kadıköy'ün mekanlarını puanlarla keşfet.",
      "site.detailsTagline": "Mekan detayları",
      "language.label": "Arayüz dili",
      "page.eyebrow": "Mekan rehberi",
      "page.heading": "Keşfetmeye değer mekanlar",
      "analyzer.eyebrow": "Yorum analizi",
      "analyzer.title": "Bir restoran yorumunu analiz et",
      "analyzer.description": "Yorumdaki lezzet, atmosfer, servis ve fiyat/değer görüşlerini incele.",
      "analyzer.inputLabel": "Restoran yorumu",
      "analyzer.placeholder": "Restoran yorumunu buraya yapıştır",
      "analyzer.button": "Yorumu Analiz Et",
      "analyzer.loading": "Analiz ediliyor...",
      "analyzer.complete": "Analiz tamamlandı.",
      "analyzer.details": "Analiz detayları",
      "analyzer.detailsSummary": "{sentences} cümle incelendi; {calls} sentiment çıkarımı yapıldı.",
      "analyzer.evidence": "{aspect} kanıtları",
      "analyzer.confidence": "Güven: {value}%",
      "analyzer.notMentioned": "Bahsedilmedi",
      "analyzer.error.empty": "Lütfen analiz etmek için bir restoran yorumu girin.",
      "analyzer.error.network": "Backend'e ulaşılamadı. FastAPI servisinin çalıştığını kontrol edin.",
      "analyzer.error.invalidJson": "Backend geçerli bir JSON yanıtı döndürmedi.",
      "analyzer.error.api": "Yorum analiz edilemedi.",
      "analyzer.error.missingAspects": "Analiz yanıtında aspect sonuçları bulunamadı.",
      "analyzer.error.unexpected": "Yorum analizi sırasında beklenmeyen bir hata oluştu.",
      "aspect.taste": "Lezzet",
      "aspect.atmosphere": "Atmosfer",
      "aspect.service": "Servis",
      "aspect.value": "Fiyat / Değer",
      "aspect.staff": "Personel",
      "aspect.decoration": "Dekorasyon",
      "sentiment.positive": "Olumlu",
      "sentiment.neutral": "Nötr",
      "sentiment.negative": "Olumsuz",
      "filters.search": "Mekan ara",
      "filters.searchPlaceholder": "Mekan ara...",
      "filters.category": "Kategori",
      "filters.allCategories": "Tümü",
      "filters.district": "İlçe",
      "filters.allDistricts": "Tüm ilçeler",
      "category.cafe": "Kafe",
      "category.bakeryCafe": "Fırın Kafe",
      "category.restaurant": "Restoran",
      "category.pub": "Pub",
      "places.category": "Kategori: {value}",
      "places.overallScore": "Genel puan: {value}",
      "places.viewDetails": "Detayları Gör",
      "places.loadError": "Mekan listesi şu anda yüklenemiyor. Yorum analizi kullanılmaya devam edebilir.",
      "details.back": "← Tüm mekanlara dön",
      "details.notFound": "Mekan bulunamadı.",
      "details.loadError": "Mekan bilgileri yüklenirken bir hata oluştu.",
      "details.scores": "Puanlar",
      "details.priceLevel": "Fiyat seviyesi: {value}",
      "details.coffeeBeans": "Kahve çekirdeği: {value}",
      "details.noCoffeeBeans": "Kahve çekirdeği bilgisi bulunmuyor.",
      "details.ourView": "Bizim görüşümüz: {value}"
    }
  };

  function readStoredLanguage() {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      return SUPPORTED_LANGUAGES.includes(stored) ? stored : DEFAULT_LANGUAGE;
    } catch (error) {
      return DEFAULT_LANGUAGE;
    }
  }

  let currentLanguage = readStoredLanguage();

  function translate(key, values) {
    const dictionary = translations[currentLanguage] || translations.en;
    let text = dictionary[key] || translations.en[key] || key;
    Object.entries(values || {}).forEach(function ([name, value]) {
      text = text.replaceAll(`{${name}}`, String(value));
    });
    return text;
  }

  function applyDocumentTranslations() {
    document.documentElement.lang = currentLanguage;
    document.querySelectorAll("[data-i18n]").forEach(function (element) {
      element.textContent = translate(element.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach(function (element) {
      element.placeholder = translate(element.dataset.i18nPlaceholder);
    });
    document.querySelectorAll("[data-i18n-aria-label]").forEach(function (element) {
      element.setAttribute(
        "aria-label",
        translate(element.dataset.i18nAriaLabel)
      );
    });
    document.querySelectorAll("[data-ui-language]").forEach(function (button) {
      const isActive = button.dataset.uiLanguage === currentLanguage;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    });

    const titleKey = document.body.dataset.titleKey;
    if (titleKey) {
      document.title = translate(titleKey);
    }
  }

  function setLanguage(language) {
    if (!SUPPORTED_LANGUAGES.includes(language)) {
      return;
    }
    currentLanguage = language;
    try {
      window.localStorage.setItem(STORAGE_KEY, language);
    } catch (error) {
      // The UI still switches when storage is unavailable.
    }
    applyDocumentTranslations();
    window.dispatchEvent(
      new CustomEvent("ui-language-changed", {detail: {language}})
    );
  }

  function initialize() {
    applyDocumentTranslations();
    document.querySelectorAll("[data-ui-language]").forEach(function (button) {
      button.addEventListener("click", function () {
        setLanguage(button.dataset.uiLanguage);
      });
    });
  }

  window.I18n = {
    initialize,
    setLanguage,
    t: translate,
    getLanguage: function () {
      return currentLanguage;
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
  } else {
    initialize();
  }
})();
