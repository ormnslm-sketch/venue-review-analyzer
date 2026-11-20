# MAMS English Neutral Candidate Review Guide

This file guides human review of `mams_neutral_candidates_en.csv`.

No candidate is accepted automatically. MAMS annotations refer to the specific aligned `source_category` and `source_polarity`, not necessarily to the overall sentiment of the complete sentence. A sentence may discuss several aspects with different polarities.

## Allowed decisions

Enter exactly one of these values in `human_decision`:

- `ACCEPT`: the source annotation maps reliably to our aspect and is genuinely neutral.
- `REJECT`: the mapping is unsuitable, the target meaning is not neutral, or the example is unsafe for training.
- `UNCERTAIN`: a second reviewer or closer context inspection is needed.

For accepted rows, fill `human_final_aspect` and `human_final_sentiment`. For this pool, the expected sentiment is `neutral`, but the reviewer must still verify it. Use `review_notes` to explain ambiguous or rejected cases. Do not remove an existing `EXACT_OVERLAP_EXISTING_DATA` marker.

## TASTE (`food` → `taste`)

Human review is mandatory. Accept only when the relevant neutral `food` annotation reliably fits our taste/product-quality concept, such as:

- taste or flavor;
- food quality;
- preparation or cooking quality.

Do not automatically accept as taste:

- portion size;
- menu variety;
- presentation;
- order logistics;
- a general food mention without a quality or taste judgement.

## ATMOSPHERE (`ambience` → `atmosphere`)

Accept when the neutral annotation concerns the ambience or environmental experience, for example music, noise, mood, room feel, or the dining environment.

Reject or mark uncertain when the relevant evidence is actually about food, staff, service logistics, location, or another aspect.

## SERVICE (`service` → `service`)

Accept when the neutral annotation concerns a service process, such as:

- wait time;
- order handling;
- serving process;
- reservations;
- seating process.

Be careful when staff behaviour is the real subject. Staff is not part of the current four-aspect baseline, so behaviour-focused examples must not be forced into `service` merely because an employee is mentioned.

## VALUE (`price` → `value`)

Accept when the neutral annotation genuinely expresses price or value information/judgement.

A price number, the word “bill,” or a payment event alone is not enough. The relevant annotation must concern price, affordability, cost, value, or a price-to-quality relationship.

## Leakage rule

Rows whose `review_notes` contains `EXACT_OVERLAP_EXISTING_DATA` exactly match a sentence already found in at least one cleaned split. They must not be added to training until the split-specific overlap report is checked. In particular, exact dev/test sentences must never be introduced into training.

## Recommended review process

1. Read `source_category`, `mapped_aspect`, and the full sentence together.
2. Judge only the aligned source annotation; do not infer a global sentence label.
3. Apply the aspect rules above.
4. Verify that the annotation is neutral rather than weakly positive or negative.
5. Record `ACCEPT`, `REJECT`, or `UNCERTAIN` and add a short note when the decision is not obvious.
