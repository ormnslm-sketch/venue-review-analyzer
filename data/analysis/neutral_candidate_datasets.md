# Neutral Data Expansion v2 — Candidate Dataset Audit

Audit date: 2026-10-30

Scope: candidate-source inspection only. No rows were added to `data/cleaned/` or `data/processed/`; no translation, synthetic generation, or model training was performed.

Decision priority: (1) license clarity, (2) restaurant-domain relevance, (3) label reliability, (4) ontology compatibility, (5) neutral diversity, and only then (6) quantity.

## Executive decision table

| Dataset | Language | Restaurant domain? | License | Gold / Silver / Synthetic | Neutral available? | Our ontology mapping quality | Recommended use |
|---|---|---:|---|---|---:|---|---|
| `justinsiow/mams-acsa` | English | Yes | Apache-2.0 | Gold, manually annotated | Yes | High for ambience/service/price; food→taste needs review | **ACCEPT WITH HUMAN REVIEW** |
| `STNM-NLPhoenix/turkish-absa` | Turkish | **Not verified; examples are multi-domain brand/entity sentiment** | Apache-2.0 | Provenance/annotation process unclear | Yes, 113,988 entity-level neutral annotations | Poor: entities are brands, not restaurant aspects | **REJECT** |
| `vinnipukh/semeval_tr_relabelled` | Turkish | Yes | MIT on the HF repository; underlying SemEval redistribution rights not independently established | Silver, automatic relabelling | Yes, but sparse after confidence filtering | Partial: useful mainly for taste | **LICENSE CHECK NEEDED** |
| SemEval-2016 Task 5 Spanish restaurant | Spanish | Yes | **LICENSE UNCLEAR** | Gold, manually annotated | Neutral is part of the polarity schema; exact raw count not audited | High category compatibility | **LICENSE CHECK NEEDED** |

## Candidate 1 — MAMS-ACSA

Sources: [Hugging Face dataset](https://huggingface.co/datasets/justinsiow/mams-acsa), [dataset card metadata](https://huggingface.co/datasets/justinsiow/mams-acsa/raw/main/README.md), [original paper](https://aclanthology.org/D19-1654/), [original repository](https://github.com/siat-nlp/MAMS-for-ABSA).

### Metadata and schema

- Language: English.
- Domain: restaurant reviews.
- License: Apache-2.0 in both the HF metadata and original repository.
- Quality: manually annotated/gold. The paper describes MAMS as a multi-aspect, multi-sentiment restaurant ABSA dataset.
- HF mirror splits: one `train` split with 3,000 sentence rows. The original MAMS project contains broader task-specific resources, so this audit's counts refer specifically to `justinsiow/mams-acsa`.
- HF schema: `text`, `category`, `polarity`; category and polarity are aligned comma-separated lists.
- Polarity values observed: `positive`, `neutral`, `negative`.
- Aspect categories observed: `ambience`, `food`, `menu`, `miscellaneous`, `place`, `price`, `service`, `staff`.

### Neutral annotation counts

| MAMS category | Candidate mapping | Neutral count | Mapping assessment |
|---|---|---:|---|
| food | taste | **1,233** | Medium/high: food is broader than taste; human review required |
| ambience | atmosphere | **48** | High |
| service | service | **121** | High |
| price | value | **125** | High |

Counts are annotation counts, not unique sentence counts. One sentence may contribute several category/polarity annotations.

### Ten neutral examples per target category

These are sentences carrying a neutral annotation for the named category. Because MAMS sentences deliberately contain several aspects with different polarities, the entire sentence may also contain strong positive or negative language about another aspect. Human review must inspect the aligned category/polarity pair rather than judging the sentence globally.

#### food — neutral

1. I like the smaller portion size for dinner.
2. The bill was surprisingly inexpensive considering we each had appetizers, an entree, dessert and drinks (alcoholic and non) we also had 3 rounds of shots for the entire table.
3. ) other food is served in too-small portions, but at least it leaves room for dessert.
4. After ordering drinks, we both decided on the Paella Vallenciana, brought out on hot plates.
5. Portions are fairly generous and the staff brings out multiple little bites and treats throughout dinner.
6. Specialty drinks--alcoholic and non--arrive in skull mugs or mini-canteens, nifty take-home souvenirs.
7. Towards the end of our meal, a server came out, apparently our orders had been double-filled.
8. Upon entering, I was impressed by the room while the food on other peoples' tables seemed enticing.
9. The beginning of the meal wasnt bad, the hostess was very nice, we got our drinks about every 10 minutes and the appetizers we good.
10. Our waitress couldn't tell us what was in the seafood special, forgot to put in our oyster order so it came at the same time as the main meals, wasn't able to provide a confident recommendation from the menu, and had to be flagged several times for drinks.

#### ambience — neutral

1. I found that the food variety was great, and the waitress was very accommodating to my vegan boyfriend describing all items' ingredients and how you may request more of what YOU like, creating a unique experience.
2. Cornelia Street looks like a Broadway set for West Side Story and the inside of Po is so cool quaint you really can't top the setting for a romantic dinner in NYC.
3. Unfortunately, the food along with the unhelpful service doesn't make up for the atmosphere.
4. I waited 15 minutes for the waiter to take the bill from me and finally I got fed up with him walking past our table I had to start waving the bill in the air and finally some other waitress took our bill.
5. The Food The menu seems quirky, but upon a closer look ingredients are familiar and made into appealing combinations, with influences from Maine to the Mediterranean.
6. employees not fighting with each other in front of clients 3 having ALL the ingredients for the meals they have on there menue.
7. I KEPT WAITING FOR THE FLAVORS IN THIS RICH LOOKING SAUCE TO TAKE OVER, BUT THEY DID NOT, THE SCALLOP WAS A LITTLE RUBBERY.
8. The fresh sliced mozzerella, the sweet roasted peppers, the hint of garlic and oil, are just some of the ingredients that produce the light otherworldly tasty pizza at Grimaldi's.
9. While the staff was kind and attentive, our waiter never once broke a smile, even when I told him it was my friend's birthday, although he did bring a dessert with a candle at the end of the meal.
10. The music is a mix of house radio as well as live band at the same time.

#### service — neutral

1. Came recommended to us, but we found the food to be so-so, the service good, but we were told we could not order desert since the table we were at had a reservation waiting.
2. worst margaritas in town,if you are going to order a battle of (white wine)they served hot.
3. The food was served promptly and was really hot.
4. The kitchen is open and surrounded by bar seating (the only kind of seating), and they have a nice set-up where the waitress stays behind the bar and the chefs deliver your food after you have watched them prepare it.
5. There is a huge line, usually a 45 minute wait; they take no reservations and no tables of more than five people.
6. Spoke with the manager about comping a round, after we had already paid for 2 rounds waiting for the table.
7. Service is one thing but a restaurant like Craft needs to have much better quality food for the price they charge.
8. If you've been out partying and have a yen for quality Korean food at 3:00am, there's a table waiting for you at Kang Suh Restaurant.
9. there's always people waiting to be seated and the chairs are not comfortable.
10. While waiting there was no room for standing and the bar staff and their drinks were horrendous.

#### price — neutral

1. Smiling servers quiz diners with movie trivia, but forget to fill their water glasses.
2. considering the prices on the menu, i'd rather go somewhere else where they know how to refill water and real chairs for everyone, something you can even get at mcdonald's.
3. *There is a pre-theatre dinner menu for early diners that is of great value.
4. The situation was verified by taking 3 dishes that were over an hour late off the bill and free desserts and coffee but the experience left most of us never wanting to go to the supposed hip cool 66 ever again.
5. When we got our food, the waitress barely said a word as she placed our take-out on the table and grabbed the check.
6. All this Asian fusion craze leaves one feeling like they should've went to an authentic Thai or Indian restaurant for probably 1/3 of the price of Spice Market.
7. Food was at best, mediocre and when the bill finally came they had nerve to add on 20% gratuity even though there was nothing stated on the menu or anywhere that gratuity was included for a party of 5 (usually a party of 6 is gratuity inclusive).
8. And once the bill was paid, the bartender kept on giving us drinks.
9. In any case, the food was quite yummy and on monday nights, they have a prix fixe $25 dinner which is definitely a deal.
10. Not ony did our waiter impose on the dessert we were trying to enjoy to take the money -- he didn't even ask us if we wanted change -- nor did he bring us back any!!!

### Verdict

**ACCEPT WITH HUMAN REVIEW.** It is the strongest English candidate: open license, restaurant relevance, gold labels, and ample neutral supply. Before inclusion, review `food→taste` and each selected annotation in context. Exclude `staff` and all non-baseline categories.

## Candidate 2 — Turkish ABSA

Source: [Hugging Face dataset and card](https://huggingface.co/datasets/STNM-NLPhoenix/turkish-absa).

### Metadata and audit

- Language: Turkish.
- License in HF metadata: Apache-2.0.
- Split: one `train` split, 200,760 rows.
- Schema: `text` plus `entities`, where every entity contains `entity` and `sentiment`.
- Sentiment labels observed: `positive`, `neutral`, `negative`.
- Entity annotations: 420,198 total; **113,988 neutral**.
- Unique entity strings: 16,843.
- Dataset source and annotation method: the card only describes customer feedback; it does not identify a named source corpus or document a gold human-annotation process.

Top entity counts demonstrate that the labels describe companies/brands rather than restaurant aspects:

| Entity | Count | Entity | Count |
|---|---:|---|---:|
| Vodafone | 5,922 | Apple | 5,628 |
| Samsung | 5,260 | Microsoft | 4,846 |
| Türk Telekom | 4,800 | Nike | 4,456 |
| Turkcell | 4,094 | Sony | 3,946 |
| Adidas | 3,850 | Google | 3,416 |
| Puma | 3,208 | Amazon | 2,946 |
| Ford | 2,894 | LG | 2,806 |
| Siemens | 2,740 | Bosch | 2,706 |

### Ontology assessment

The `entity` field is not an aspect category. A brand such as Vodafone, Apple, Nike, or Ford cannot be mapped reliably to `taste`, `atmosphere`, `service`, or `value`. Some sentences happen to mention service or price, but deriving our aspects from sentence wording would require a new annotation step and cannot be treated as an existing reliable mapping. Restaurant-domain status is therefore **not verified** and the observed data is clearly multi-domain.

### Verdict

**REJECT.** The 113,988 neutral annotations do not solve our restaurant aspect gap. Quantity cannot compensate for domain mismatch, unclear annotation provenance, and incompatible entity labels.

## Candidate 3 — Turkish restaurant SemEval relabelled

Source: [Hugging Face dataset and card](https://huggingface.co/datasets/vinnipukh/semeval_tr_relabelled). Underlying task reference: [SemEval-2016 Task 5](https://aclanthology.org/S16-1002/).

### Metadata and parse audit

- Language/domain: Turkish restaurant reviews from SemEval-2016 Task 5.
- HF repository license metadata: MIT.
- Split: one `train` split, 1,232 rows.
- Schema: `review_id`, `sentence_id`, `text`, string-form `label`, and `confidence`.
- Quality: silver-labelled. The dataset card states that an automatic system selected labels and that the output was not manually checked; median confidence is reported as 0.42.
- Successfully parsed list rows: 1,134.
- Unparseable/non-list labels: **98**, comprising 54 `[]` rows plus 44 `OutOfScope` or note-contaminated strings.
- Neutral annotations before confidence filtering: **96**.

### Neutral annotations by category and confidence

| Category | All neutral | confidence ≥ 0.7 | confidence ≥ 0.8 | Candidate mapping |
|---|---:|---:|---:|---|
| FOOD#QUALITY | 44 | 11 | 10 | taste |
| RESTAURANT#GENERAL | 20 | 3 | 3 | excluded |
| SERVICE#GENERAL | 7 | 2 | 0 | service |
| RESTAURANT#PRICES | 7 | 0 | 0 | value |
| FOOD#PRICES | 6 | 1 | 0 | value |
| FOOD#STYLE_OPTIONS | 4 | 0 | 0 | excluded |
| DRINKS#QUALITY | 2 | 0 | 0 | taste |
| LOCATION#GENERAL | 2 | 0 | 0 | excluded |
| AMBIENCE#GENERAL | 2 | 0 | 0 | atmosphere |
| RESTAURANT#MISCELLANEOUS | 1 | 0 | 0 | excluded |
| DRINKS#PRICES | 1 | 0 | 0 | value |
| **Total** | **96** | **17** | **13** | — |

At confidence ≥ 0.7, our usable mapping candidates are: taste 11, atmosphere 0, service 2, value 1. At confidence ≥ 0.8 they fall to: taste 10, atmosphere 0, service 0, value 0. The remaining high-confidence neutral items are excluded `RESTAURANT#GENERAL` annotations.

### License assessment and verdict

The derivative HF repository declares MIT, but this audit did not find an explicit statement proving that the underlying SemEval restaurant text and annotations may be redistributed or relicensed under MIT. Therefore: **LICENSE CHECK NEEDED**. If source rights are confirmed, use only with human review because labels are silver and high-confidence coverage is sparse. This is the most promising Turkish candidate in this audit, but it is not yet approved for training.

## Candidate 4 — SemEval-2016 Task 5 Spanish restaurant

Sources: [official data-and-tools page](https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools), [official task paper](https://aclanthology.org/S16-1002/).

### Verified properties

- Language/domain: Spanish, restaurant domain.
- Quality: gold/manual ABSA annotations.
- Annotation tuple: aspect category `E#A`, optional opinion target expression, and polarity.
- Polarity inventory includes `positive`, `negative`, and `neutral`.
- Restaurant aspect categories include `RESTAURANT#GENERAL`, `RESTAURANT#PRICES`, `RESTAURANT#MISCELLANEOUS`, `FOOD#PRICES`, `FOOD#QUALITY`, `FOOD#STYLE_OPTIONS`, `DRINKS#PRICES`, `DRINKS#QUALITY`, `DRINKS#STYLE_OPTIONS`, `AMBIENCE#GENERAL`, `SERVICE#GENERAL`, and `LOCATION#GENERAL`.
- Official task-paper train size for Spanish restaurant Subtask 1: **627 texts/reviews, 2,070 sentences, 2,720 opinion tuples**.

The target ontology is structurally compatible with our existing mapping for food/drinks quality→taste, ambience→atmosphere, service→service, and price categories→value.

### License gate

The official task pages link data through META-SHARE but this audit did not find a clear, explicit open license and redistribution statement governing repository inclusion and training reuse. Consequently:

- **STATUS = LICENSE UNCLEAR**
- Neutral polarity is verified as part of the schema, but the exact raw neutral count was **not audited**.
- The dataset was not downloaded, copied into the repository, or added to training.

### Verdict

**LICENSE CHECK NEEDED.** It is likely the best Spanish candidate on domain, annotation quality, and ontology compatibility, but legal/redistribution clarity is a hard prerequisite.

## Recommended next source by language

- **English:** MAMS-ACSA — proceed to a human-review sampling/import plan, especially for `food→taste`.
- **Turkish:** Turkish SemEval relabelled is the best candidate among those audited, but first verify rights for the underlying SemEval data; then manually review high-confidence silver labels. It cannot fill atmosphere/service/value gaps by itself.
- **Spanish:** SemEval-2016 Spanish restaurant — first obtain an explicit license/redistribution determination from the official source or rights holder. Until then, no audited Spanish source is approved.

## Final conclusion

Only MAMS is ready for the next controlled curation step, and even there the mapping should be human-reviewed. The Turkish multi-domain ABSA dataset is rejected despite its large neutral count. Both SemEval-derived candidates remain outside training until source-data rights are clarified; the Turkish derivative also requires label-quality review.
