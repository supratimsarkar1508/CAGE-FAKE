# Update decision_rule.py
with open('CAGE-FAKE/src/decision_rule.py', 'r', encoding='utf-8') as f:
    d_code = f.read()

old_d_block = '''        # --- Credibility & High-Similarity Evidence Overrides ---
        credibility_index = float(multi_view_summary.get("credibility_index", 50.0))
        claim_verifs = evidence_summary.get("claim_verifications", [])
        high_sim_supported = [
            c for c in claim_verifs
            if c.get("status") == "Supported" and float(c.get("similarity", c.get("dense_score", 0.0))) >= 0.65
        ]
        n_high_sim = len(high_sim_supported)

        # Evidence Override condition: >=2 supported claims (sim >= 0.65) AND credibility index >= 80
        evidence_override_active = (
            n_high_sim >= 2
            and credibility_index >= 80.0
            and not has_contra
            and not has_num_conflict
        )'''

new_d_block = '''        # --- Credibility & High-Similarity Evidence Overrides ---
        credibility_index = float(multi_view_summary.get("credibility_index", 50.0))
        sens_discipline = float(multi_view_summary.get("sensationalism_discipline", credibility_index))
        has_high_credibility = (credibility_index >= 75.0 or sens_discipline >= 80.0)
        claim_verifs = evidence_summary.get("claim_verifications", [])
        high_sim_supported = [
            c for c in claim_verifs
            if c.get("status") == "Supported" and float(c.get("similarity", c.get("dense_score", 0.0))) >= 0.60
        ]
        n_high_sim = len(high_sim_supported)

        # Evidence Override condition: >=2 supported claims AND high credibility/sensationalism discipline
        evidence_override_active = (
            n_high_sim >= 2
            and has_high_credibility
            and not has_contra
            and not has_num_conflict
        )'''

if old_d_block in d_code:
    d_code = d_code.replace(old_d_block, new_d_block, 1)

with open('CAGE-FAKE/src/decision_rule.py', 'w', encoding='utf-8') as f:
    f.write(d_code)
print('decision_rule.py updated!')

# Update predict.py
with open('CAGE-FAKE/src/predict.py', 'r', encoding='utf-8') as f:
    p_code = f.read()

old_p_block = '''        claim_verifs = evidence_res.get("claim_verifications", [])
        high_sim_supported = [
            c for c in claim_verifs
            if c.get("status") == "Supported" and float(c.get("similarity", c.get("dense_score", 0.0))) >= 0.65
        ]
        n_high_sim = len(high_sim_supported)
        has_contra = evidence_res.get("has_any_contradiction", False) or evidence_res.get("has_numeric_conflict", False)
        evidence_override_active = (n_high_sim >= 2 and cred_index >= 80.0 and not has_contra)'''

new_p_block = '''        sens_discipline = float(credibility_info["scores"].get("Sensationalism Discipline", cred_index))
        has_high_credibility = (cred_index >= 75.0 or sens_discipline >= 80.0)
        claim_verifs = evidence_res.get("claim_verifications", [])
        high_sim_supported = [
            c for c in claim_verifs
            if c.get("status") == "Supported" and float(c.get("similarity", c.get("dense_score", 0.0))) >= 0.60
        ]
        n_high_sim = len(high_sim_supported)
        has_contra = evidence_res.get("has_any_contradiction", False) or evidence_res.get("has_numeric_conflict", False)
        evidence_override_active = (n_high_sim >= 2 and has_high_credibility and not has_contra)'''

if old_p_block in p_code:
    p_code = p_code.replace(old_p_block, new_p_block, 1)

old_mvr_block = '''        multi_view_res = {
            "mean_p_fake": p_cal_val,
            "view_disagreement": dis_val,
            "credibility_index": cred_index,'''

new_mvr_block = '''        multi_view_res = {
            "mean_p_fake": p_cal_val,
            "view_disagreement": dis_val,
            "credibility_index": cred_index,
            "sensationalism_discipline": sens_discipline,'''

if old_mvr_block in p_code:
    p_code = p_code.replace(old_mvr_block, new_mvr_block, 1)

with open('CAGE-FAKE/src/predict.py', 'w', encoding='utf-8') as f:
    f.write(p_code)
print('predict.py updated!')

