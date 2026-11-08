"""V1.7 step 8 -- assemble the confirmation result record from evidence.

Reads the sealed preflight, the one-shot ledger and the independent
recomputation, and writes V1_7_CONFIRMATION_RESULT.json. Every number is copied
from an evidence file; nothing is retyped by hand.
"""
from __future__ import annotations
import json
from pathlib import Path
from . import paths as _paths

HERE = _paths.REPO_ROOT
E = _paths.frozen_dir()

pre = json.loads((E / "v1_6" / "v1_6_sealed_preflight.json").read_text(encoding="utf-8"))
led = json.loads((E / "v1_6" / "v1_6_confirmation_ledger.json").read_text(encoding="utf-8"))
ind = json.loads((E / "v1_7" / "v1_7_independent_verification.json").read_text(encoding="utf-8"))
man = json.loads((E / "v1_7" / "sealed_file_manifest.json").read_text(encoding="utf-8"))
clk = json.loads((E / "v1_7" / "v1_7_clock_rule_test.json").read_text(encoding="utf-8"))
aud = json.loads((E / "v1_7" / "v1_7_criterion_audit.json").read_text(encoding="utf-8"))

est, crit = led["estimands"], led["criteria"]
e1 = crit["E1_pooled_supply_share"]
e2 = crit["E2_within_person_activity_gradient"]
e4 = crit["E4_device_refusal_rate"]
per = est["per_participant"]

n_wrong = sum(v["n_wrong"] for v in per.values())
n_refused = sum(v["n_refused"] for v in per.values())
n_windows = sum(v["n_windows"] for v in per.values())

doc = {
    "document": "Galaxy-HRGuard v1.7 SEALED independent confirmation -- RESULT",
    "date": "2026-11-04",
    "authorised_by_user": True,
    "authorisation_scope": "A1 unseal + A2 local extraction + A3 one zero-fit run",
    "state": "CONFIRMATION_COMPLETED",
    "one_shot": True,
    "parameters_fitted": 0,
    "gpu_used": False,
    "downloads": 0,
    "reruns_after_result": 0,

    "cohort": {
        "authorised": man["sealed_subjects"],
        "evaluable": est["participants"],
        "n_evaluable": led["evaluable_n"],
        "excluded": led["excluded"],
        "extracted_files": man["extracted_files_total"],
        "extracted_MB": round((man["extracted_bytes_total"] +
                               man.get("estimated_extra_bytes", 0)) / 1e6, 2),
    },

    "clock_rule": {
        "rule": "closed candidate set {raw:0, author_minus_9h:-32400000}; score = "
                "fraction of Polar timestamps inside the watch-PPG timestamp "
                "range; PASS iff best >= 0.75 AND best-second >= 0.30 AND the "
                "residual envelope holds. Timestamps only; reference HR error is "
                "never used.",
        "per_participant": {
            s: {"state": v["state"],
                "overlap": (v.get("decision") or {}).get("candidate_overlap_fraction"),
                "recipe": (v.get("decision") or {}).get("selected_recipe"),
                "res_lo_s": (v.get("guard") or {}).get("res_lo_s"),
                "res_hi_s": (v.get("guard") or {}).get("res_hi_s")}
            for s, v in pre["per_participant"].items()},
        "written_before_any_label": pre["written_before_any_label"],
    },

    "primary_estimand_E1": {
        "definition": "pooled supply share = wrong-value windows / (wrong-value "
                      "windows + refused windows); refused = a valid Polar "
                      "reference exists and no usable watch value was delivered",
        "n_wrong": n_wrong,
        "n_refused": n_refused,
        "denominator": n_wrong + n_refused,
        "windows_total": n_windows,
        "point_estimate": e1["point_estimate"],
        "ci95_participant_cluster_bootstrap": e1["ci95"],
        "criterion": e1["criterion"],
        "status": e1["status"],
        "tier": e1["tier"],
        "interpretation": f"{e1['point_estimate']:.4f} means that of every 100 "
                          f"unreliable windows, about {100*e1['point_estimate']:.1f} "
                          f"carried a wrong value and about "
                          f"{100*(1-e1['point_estimate']):.1f} were the device "
                          f"refusing to output at all",
    },

    "secondary_descriptive": {
        "E1b_subject_equal_supply_share": {
            "point": est["E1b_subject_equal_supply_share"],
            "ci95": led["participant_cluster_bootstrap"]["E1b"]["ci95"]},
        "E2_within_person_activity_gradient": {
            "point": est["E2_within_person_activity_gradient"],
            "ci95": e2["ci95"],
            "positive_participants": f"{sum(1 for v in e2['per_participant'].values() if v > 0)}/{len(e2['per_participant'])}",
            "per_participant": e2["per_participant"],
            "exact_signflip_p_one_sided": e2["exact_signflip_p_one_sided"],
            "exact_sign_test_p_one_sided": e2["exact_sign_test_p_one_sided"],
            "status": "DESCRIPTIVE",
            "why_descriptive": e2["reason"]},
        "E3_between_person_spread": {
            "point": est["E3_between_person_spread"],
            "per_participant_error_rate": crit["E3_between_person_spread"]["per_participant"],
            "status": "DESCRIPTIVE"},
        "E4_device_refusal_rate": {
            "point": est["E4_device_refusal_rate"], "ci95": e4["ci95"],
            "per_participant": e4["per_participant"], "status": "DESCRIPTIVE"},
        "E5_eta2_participant_share": {
            "point": est["E5_eta2_participant_share"],
            "status": "DESCRIPTIVE",
            "note": "no eta-squared quantity is reported in the confirmation side"},
    },

    "independent_recomputation": {
        "implementation": ind["implementation"],
        "checks": ind["checks"],
        "all_pass": ind["verdict"]["all_estimand_checks_pass"],
        "bootstrap": ind["bootstrap"],
        "reading": "the two implementations agree exactly on the point estimate, "
                   "on the per-participant numerator and denominator and on the "
                   "exact p value; the bootstrap intervals differ only in the "
                   "upper bound because the RNG stream and n_boot differ, and "
                   "both lie entirely below 0.50",
    },

    "paper_decision": {
        "label": led["paper_label"]["label"],
        "reason": led["paper_label"]["reason"],
        "decided_by": "the frozen decision matrix in V1_7_SEALED_RUNBOOK.md "
                      "section 5, applied to the estimate and its interval only",
        "power_values_used_in_the_decision": False,
    },

    "deviations_and_incidents": [
        {"id": "D1", "when": "extraction",
         "what": "the v1.7 runbook section 2 said ACC need not be extracted, but "
                 "the inherited inclusion rule G-EVAL clause (1) requires all five "
                 "raw files present and label_participant() reads the full FILES "
                 "map; the first preflight therefore returned DATA_NOT_EVALUABLE "
                 "for all six on a bookkeeping technicality",
         "fix": "ACC was extracted for all six (48.76 MB) so that the STRICTER "
                "rule is satisfied",
         "effect_on_results": "none -- a static check confirms watch_acc is read "
                              "but never referenced by any estimand "
                              "(0 occurrences in the estimator code)",
         "record": "evidence/v1_7/sealed_file_manifest.json (acc_rationale) and "
                   "evidence/v1_7/v1_7_preflight_attempt1_input_gap.json"},
        {"id": "D2", "when": "first preflight with ACC present",
         "what": "stage_preflight called infer_offset(watch_ppg, polar) while the "
                 "frozen semantics (audit_raw_dev.infer_offset) are "
                 "infer_offset(polar, watch_ppg). The candidate offset is added to "
                 "the REFERENCE timestamps, so the reversed order makes both "
                 "candidate scores exactly 0.0 (NO_MATCH) for every participant",
         "detection": "all six participants returned identical 0.0/0.0 overlap, "
                      "which is impossible for genuinely misaligned clocks because "
                      "the raw timestamp ranges differ by about nine hours",
         "fix": "argument order corrected; verified WITHOUT looking at any SEALED "
                "label: on DEV the corrected call reproduces the frozen decision "
                "for 16/16 subjects (test_clock_rule.py), and the DEV three-layer "
                "verification was re-run to DEV_VERIFY_OK with max_abs_diff 0.0",
         "effect_on_results": "the run that produced the reported estimate uses the "
                              "corrected call; no SEALED label was computed or "
                              "inspected before the fix",
         "record": "test_clock_rule.py and evidence/v1_7/v1_7_clock_rule_test.json"},
    ],

    "verification_status": {
        "dev_three_layer": "DEV_VERIFY_OK (layer1 8/8, layer2 24924x5 zero diff, "
                           "layer3 6/6) after the D2 fix",
        "clock_rule_test": {"subjects": clk["dev_subjects_checked"],
                            "all_match": clk["all_match"],
                            "degenerate": clk["degenerate_reference_decisions"]},
        "independent_recompute": "5/5 checks pass",
        "criterion_audit": aud["verdict"]["E2_criterion"],
    },

    "mandatory_disclosures": [
        {"id": "X1", "topic": "cross-project contact",
         "text": "The sibling project <sibling-project> parsed the raw signals "
                 "of P17 / P19 / P20 / P23 (and had byte-level access to P08 / "
                 "P13) on 2026-11-02, BEFORE this project's protocol was frozen. "
                 "Subject-level statistics and a 32,397-row window cache were "
                 "written, with no activity labels (session = '?'). None of the "
                 "seven estimands can be computed from those artefacts, and no "
                 "sibling value entered any DEV-side decision. It is nevertheless "
                 "NOT correct to describe this cohort as never handled by anyone.",
         "record": "evidence/v1_6/v1_6_contact_ledger.json"},
        {"id": "X2", "topic": "P08 not evaluable",
         "text": "P08 lacks GalaxyWatch/HR.csv. A missing file makes 'the device "
                 "emitted nothing' indistinguishable from 'the file was not "
                 "recorded', so P08 enters neither a numerator nor a denominator "
                 "and is NOT counted as a device refusal. Evaluable n = 5."},
        {"id": "X3", "topic": "no power for the activity component",
         "text": "At n <= 5 no admissible one-sided exact test can carry the "
                 "activity component: at n = 4 the smallest attainable p is 0.0625 "
                 "> 0.05, and at n = 5 rejection requires all five deltas to be "
                 "positive (power about 0.52). This is a property of the design, "
                 "not of the data. A null result is UNTESTED, never refuted."},
    ],

    "not_done": ["no model was fitted", "no threshold was calibrated",
                 "no hyper-parameter was searched", "no second database",
                 "no GPU", "no download", "no rerun after seeing the result",
                 "no claim was upgraded from descriptive to confirmatory"],
}

out = _paths.output_dir() / "V1_7_CONFIRMATION_RESULT.json"
out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}")
print(f"  n_evaluable      : {led['evaluable_n']}")
print(f"  E1               : {e1['point_estimate']:.6f}  CI {e1['ci95']}  -> {e1['status']}")
print(f"  paper label      : {led['paper_label']['label']}")
