"""Reader-facing research terms for stable machine-readable metric identifiers."""

TASK_LABELS = {
    'L3': 'Ego-action branching',
    'L4': 'Target-actor intervention',
}
METRIC_LABELS = {
    'Causal_Foresight_F1': 'Branch-set reference F1',
    'Counterfactual_Alignment_Score': 'Counterfactual Alignment Score (CAS)',
    'CTMS': 'CTMS (supplementary reference aggregate)',
    'L4_control_similarity': 'Control-chain reference agreement',
    'L4_treated_similarity': 'Treated-chain reference agreement',
    'L3_valid_only': 'Branch-set reference F1 on valid outputs',
    'L4_valid_only': 'CAS on valid pairs',
    'L4_control_valid_only': 'Control-chain reference agreement on valid chains',
    'L4_treated_valid_only': 'Treated-chain reference agreement on valid chains',
}


def metric_label(identifier):
    """Keep stored keys intact while rendering task names in reader-facing tables."""
    if identifier in METRIC_LABELS:
        return METRIC_LABELS[identifier]
    if '/' in identifier:
        metric, suffix = identifier.split('/', 1)
        return f'{metric_label(metric)} / {suffix.replace("_", " ")}'
    for code, task in TASK_LABELS.items():
        if identifier.startswith(code + '_'):
            return f'{task}: {identifier[len(code) + 1:].replace("_", " ")}'
    return identifier.replace('_', ' ')
