const detailsContainer = document.querySelector("#place-details");
const searchParams = new URLSearchParams(window.location.search);
const selectedId = Number(searchParams.get("id"));
let selectedPlace = null;
let detailState = "loading";

function t(key, values) {
  return window.I18n.t(key, values);
}

function renderPlaceDetails() {
  if (detailState === "not-found") {
    detailsContainer.innerHTML = `<p>${t("details.notFound")}</p>`;
    return;
  }
  if (detailState === "error") {
    detailsContainer.innerHTML = `<p>${t("details.loadError")}</p>`;
    return;
  }
  if (!selectedPlace) {
    return;
  }

  const coffeeBeansText =
    selectedPlace.coffeeBeans === null
      ? t("details.noCoffeeBeans")
      : selectedPlace.coffeeBeans;

  detailsContainer.innerHTML = `
    <article class="place-card">
      <h2>${selectedPlace.name}</h2>
      <p>${selectedPlace.neighborhood}, ${selectedPlace.district}</p>
      <p>${t("places.category", {value: selectedPlace.category})}</p>
      <p class="overall-score">${t("places.overallScore", {value: selectedPlace.overallScore})}</p>

      <h3>${t("details.scores")}</h3>
      <ul>
        <li>${t("aspect.taste")}: ${selectedPlace.scores.taste}</li>
        <li>${t("aspect.atmosphere")}: ${selectedPlace.scores.atmosphere}</li>
        <li>${t("aspect.service")}: ${selectedPlace.scores.service}</li>
        <li>${t("aspect.staff")}: ${selectedPlace.scores.staff}</li>
        <li>${t("aspect.value")}: ${selectedPlace.scores.value}</li>
        <li>${t("aspect.decoration")}: ${selectedPlace.scores.decoration}</li>
      </ul>

      <p>${t("details.priceLevel", {value: selectedPlace.priceLevel})}</p>
      <p>${t("details.coffeeBeans", {value: coffeeBeansText})}</p>
      <p>${t("details.ourView", {value: selectedPlace.aiSummary})}</p>
    </article>
  `;
}

async function loadPlaceDetails() {
  try {
    const response = await fetch(
      `http://127.0.0.1:8000/places/${selectedId}`
    );

    if (response.status === 404) {
      detailState = "not-found";
      renderPlaceDetails();
      return;
    }

    if (!response.ok) {
      throw new Error(`HTTP hatası: ${response.status}`);
    }

    selectedPlace = await response.json();
    detailState = "loaded";
    renderPlaceDetails();
  } catch (error) {
    detailState = "error";
    renderPlaceDetails();
  }
}

window.addEventListener("ui-language-changed", renderPlaceDetails);
loadPlaceDetails();
