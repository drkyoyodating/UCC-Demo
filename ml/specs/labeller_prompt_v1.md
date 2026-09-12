# Blind labeller prompt — UCC borrower relevance, label_policy_v1

prompt_version: labeller_prompt_v1

This file has two parts. Everything above the line `=== BRIEF ===` is for the ORCHESTRATOR — the
Claude session executing Plan A — and the labeller never sees it. Everything below that line is the
brief itself.

## How the orchestrator runs a blind pass (contract K14)

1. `./.venv-ml/bin/python -m ucc_ml.cli labeller-brief --config ml/configs/v1.yaml --round <round> --pass <a|b> --part <N>`
   renders one brief: the text below `=== BRIEF ===`, then the full text of
   `ml/specs/label_policy_v1.md`, then the queue chunk `queue_<round>_part_<NNN>.csv` inline as CSV
   (columns `case_id`, `borrower_name_raw`, `lender_names_raw`, `city`, `state` — nothing else).
2. One fresh subagent per chunk per pass. The orchestrator pastes the rendered brief verbatim as that
   subagent's whole prompt. The subagent has no tools: no file reads, no shell, no web search, no
   other agents. It returns its rows as structured output matching the schema below.
3. Pass A and pass B are separate agents. A pass-B agent never sees pass A's output, and no agent sees
   another chunk, a labels file, the queue key, the candidates, or any rule or model output.
4. The orchestrator — never the labeller — saves the structured output verbatim as JSON and runs
   `./.venv-ml/bin/python -m ucc_ml.cli write-raw-labels --config ml/configs/v1.yaml --round <round> --pass <a|b> --part <N> --structured <file.json>`,
   which validates the rows and writes `ml/data/labels/v1/raw/labeller_output_<round>_pass_<a|b>_part_<NNN>.csv`
   with the header `case_id,label,reason_code,reason`. When the rows are rejected the orchestrator
   discards that output and runs a fresh subagent for the same chunk and pass; labels are never
   edited by hand.
5. At most five labeller subagents run at the same time.

Structured output schema (`ucc_ml.labeling.LABELLER_OUTPUT_SCHEMA`; a test keeps the two identical):

```json
{
  "additionalProperties": false,
  "properties": {
    "rows": {
      "items": {
        "additionalProperties": false,
        "properties": {
          "case_id": {"pattern": "^[0-9a-f]{64}$", "type": "string"},
          "label": {"enum": ["RELEVANT", "NOT_RELEVANT", "INSUFFICIENT_EVIDENCE"], "type": "string"},
          "reason": {"maxLength": 300, "minLength": 1, "type": "string"},
          "reason_code": {
            "enum": ["R_LENDER_NAMED_MAKER", "R_BORROWER_EQUIPMENT_WORD", "R_BORROWER_TRADE_WORD",
                     "R_BORROWER_OTHER_EVIDENCE", "N_OTHER_INDUSTRY_EXPLICIT", "N_WORD_IS_DIFFERENT_SENSE",
                     "I_GENERIC_NAME", "I_PERSONAL_NAME", "I_AMBIGUOUS_WORD", "I_UNREADABLE"],
            "type": "string"
          }
        },
        "required": ["case_id", "label", "reason_code", "reason"],
        "type": "object"
      },
      "type": "array"
    }
  },
  "required": ["rows"],
  "type": "object"
}
```

=== BRIEF ===

You are a blind labeller for a dataset of UCC secured-lending filings from Colorado and Connecticut.
You label every row of ONE queue chunk under the written label policy that follows this brief. You are
one of two independent passes; you do not know what the other pass decided, and a founder reviews the
work.

## What you have

Everything you may use is in this message: this brief, the label policy, and the queue chunk as CSV
with the columns `case_id`, `borrower_name_raw`, `lender_names_raw` (several lenders are joined with
` | `; blank means no lender was recorded), `city`, `state`.

## Rules

- You have no tools and must not ask for any. Do not look anything up: No Google, no Secretary of
  State search, no maps, no memory of a particular firm's website. Decide each row from its five
  columns and the policy alone.
- Label every row independently, in the order given. Do not revisit an earlier row after seeing a
  later one, and do not search for duplicates (some rows are repeated on purpose under a different
  `case_id`).
- Every row gets exactly one `label`, exactly one `reason_code` from that label's family, and a
  one-sentence `reason` (at most 300 characters, no URL, no line breaks) that quotes the deciding
  word(s).

Allowed values — anything else is rejected:
- `label`: `RELEVANT`, `NOT_RELEVANT`, `INSUFFICIENT_EVIDENCE`
- with `RELEVANT`: `R_LENDER_NAMED_MAKER`, `R_BORROWER_EQUIPMENT_WORD`, `R_BORROWER_TRADE_WORD`, `R_BORROWER_OTHER_EVIDENCE`
- with `NOT_RELEVANT`: `N_OTHER_INDUSTRY_EXPLICIT`, `N_WORD_IS_DIFFERENT_SENSE`
- with `INSUFFICIENT_EVIDENCE`: `I_GENERIC_NAME`, `I_PERSONAL_NAME`, `I_AMBIGUOUS_WORD`, `I_UNREADABLE`

## Your answer

Return structured output only: an object `{"rows": [...]}` with exactly one object per queue row, in the
same order, each `{"case_id": ..., "label": ..., "reason_code": ..., "reason": ...}` with the `case_id`
copied exactly. No extra rows, no missing rows, no commentary.
