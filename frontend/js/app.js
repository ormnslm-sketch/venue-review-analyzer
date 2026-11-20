let places = [];
let placesLoadFailed = false;

function t(key, values) {
  return window.I18n.t(key, values);
}

function renderPlaces(placeList) {
  const placesContainer = document.querySelector("#places-container");
  placesContainer.innerHTML = "";

  placeList.forEach(function (place) {
    const card = document.createElement("article");
    card.classList.add("place-card");

    card.innerHTML = `
      <h3>${place.name}</h3>
      <p>${place.neighborhood}, ${place.district}</p>
      <p>${t("places.category", {value: place.category})}</p>
      <p class="overall-score">${t("places.overallScore", {value: place.overallScore})}</p>
      <ul>
        <li>${t("aspect.taste")}: ${place.scores.taste}</li>
        <li>${t("aspect.atmosphere")}: ${place.scores.atmosphere}</li>
        <li>${t("aspect.service")}: ${place.scores.service}</li>
      </ul>
      <a href="details.html?id=${place.id}">${t("places.viewDetails")}</a>
    `;

    placesContainer.appendChild(card);
  });
}

const searchInput = document.querySelector("#search-input");
const categoryFilter = document.querySelector("#category-filter");
const districtFilter = document.querySelector("#district-filter");

function filterPlaces() {
  if (placesLoadFailed) {
    document.querySelector("#places-container").innerHTML =
      `<p class="places-error">${t("places.loadError")}</p>`;
    return;
  }
  const searchText = searchInput.value.toLowerCase();
  const selectedCategory = categoryFilter.value;
  const selectedDistrict = districtFilter.value;

  const filteredPlaces = places.filter(function (place) {
    const searchMatches = place.name.toLowerCase().includes(searchText);
    const categoryMatches =
      selectedCategory === "all" || place.category === selectedCategory;
    const districtMatches =
      selectedDistrict === "all" || place.district === selectedDistrict;

    return searchMatches && categoryMatches && districtMatches;
  });

  renderPlaces(filteredPlaces);
}

async function loadPlaces() {
  const placesContainer = document.querySelector("#places-container");
  try {
    const response = await fetch("http://127.0.0.1:8000/places");

    if (!response.ok) {
      throw new Error(`HTTP hatası: ${response.status}`);
    }

    places = await response.json();
    placesLoadFailed = false;
    renderPlaces(places);
  } catch (error) {
    placesLoadFailed = true;
    placesContainer.innerHTML =
      `<p class="places-error">${t("places.loadError")}</p>`;
  }
}

searchInput.addEventListener("input", filterPlaces);
categoryFilter.addEventListener("change", filterPlaces);
districtFilter.addEventListener("change", filterPlaces);
window.addEventListener("ui-language-changed", filterPlaces);

loadPlaces();
