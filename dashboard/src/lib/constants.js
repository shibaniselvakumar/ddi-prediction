export const FALLBACK_DRUGS = [
  { id: '11289', name: 'Warfarin' },
  { id: '5640',  name: 'Ibuprofen' },
  { id: '6809',  name: 'Metformin' },
  { id: '2554',  name: 'Aspirin' },
  { id: '41493', name: 'Simvastatin' },
  { id: '3033',  name: 'Heparin' },
  { id: '9908',  name: 'Clarithromycin' },
]

export const FALLBACK_PAIRS = [
  { drug1_name: 'Warfarin',     drug2_name: 'Ibuprofen',       pair_max_prr: 2.91, pair_avg_prr: 1.77 },
  { drug1_name: 'Aspirin',      drug2_name: 'Heparin',         pair_max_prr: 2.43, pair_avg_prr: 1.52 },
  { drug1_name: 'Simvastatin',  drug2_name: 'Clarithromycin',  pair_max_prr: 2.12, pair_avg_prr: 1.41 },
  { drug1_name: 'Metformin',    drug2_name: 'Contrast dye',    pair_max_prr: 1.98, pair_avg_prr: 1.23 },
  { drug1_name: 'Lisinopril',   drug2_name: 'Potassium',       pair_max_prr: 1.84, pair_avg_prr: 1.18 },
  { drug1_name: 'Digoxin',      drug2_name: 'Amiodarone',      pair_max_prr: 1.77, pair_avg_prr: 1.09 },
  { drug1_name: 'Ciprofloxacin',drug2_name: 'Theophylline',    pair_max_prr: 1.65, pair_avg_prr: 1.01 },
]

export const FALLBACK_RESULT = {
  probability: 0.82,
  confidence:  0.79,
  risk:        'High',
  agreement:   '3/3',
  drugAName:   'Warfarin',
  drugBName:   'Ibuprofen',
  perModel:    { logreg: 0.80, random_forest: 0.84, xgboost: 0.82 },
  shap:        { local_top: [] },
}

export const FALLBACK_SHAP = [
  { feature: 'pair_max_prr',       shap: 0.213 },
  { feature: 'pair_avg_prr',       shap: 0.148 },
  { feature: 'shared_side_effects',shap: 0.116 },
  { feature: 'pair_snr',           shap: 0.082 },
  { feature: 'pathway_overlap',    shap: 0.057 },
  { feature: 'co_report_count',    shap: 0.041 },
]

export const FALLBACK_MODEL_ROWS = [
  { name: 'Random Forest',      score: 0.89 },
  { name: 'XGBoost',            score: 0.87 },
  { name: 'Logistic Regression',score: 0.81 },
]

export const NAV_ITEMS = [
  { key: 'predict', label: 'Risk Prediction',  glyph: '⬡' },
  { key: 'explain', label: 'Explainability',   glyph: '⬢' },
  { key: 'network', label: 'Signal Network',   glyph: '⬣' },
  { key: 'dataset', label: 'Data Explorer',    glyph: '⬤' },
  { key: 'ops',     label: 'Operations',       glyph: '◈' },
]

export const RISK_META = {
  High:     { color: 'var(--red)',   cls: 'badge--high',     label: 'HIGH RISK'  },
  Moderate: { color: 'var(--amber)', cls: 'badge--moderate', label: 'MODERATE'   },
  Low:      { color: 'var(--green)', cls: 'badge--low',      label: 'LOW RISK'   },
}

export const getRiskMeta = (level) =>
  RISK_META[level] ?? RISK_META.Moderate
