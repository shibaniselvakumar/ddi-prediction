export const mockPrediction = {
  drug_a: 'Warfarin',
  drug_b: 'Ibuprofen',
  probability: 0.87,
  risk: 'high',
  confidence: 0.82,
  agreement: 3,
  summary:
    'Strong pharmacodynamic overlap and reported bleeding events increase the interaction risk. Consider alternative analgesics or monitor INR closely.',
}

export const mockFeatureImportance = [
  { name: 'Shared side effects', value: 0.32 },
  { name: 'ATC proximity', value: 0.24 },
  { name: 'Target similarity', value: 0.18 },
  { name: 'Enzyme overlap', value: 0.14 },
  { name: 'Transporter overlap', value: 0.12 },
]

export const mockShap = [
  { name: 'Shared side effects', value: 0.21 },
  { name: 'ATC proximity', value: 0.15 },
  { name: 'Target similarity', value: 0.08 },
  { name: 'Enzyme overlap', value: -0.03 },
  { name: 'Transporter overlap', value: -0.05 },
]

export const mockRoc = [
  { fpr: 0, tpr: 0 },
  { fpr: 0.05, tpr: 0.62 },
  { fpr: 0.1, tpr: 0.78 },
  { fpr: 0.2, tpr: 0.88 },
  { fpr: 1, tpr: 1 },
]

export const mockPr = [
  { recall: 0, precision: 1 },
  { recall: 0.2, precision: 0.94 },
  { recall: 0.4, precision: 0.9 },
  { recall: 0.6, precision: 0.83 },
  { recall: 0.8, precision: 0.74 },
  { recall: 1, precision: 0.6 },
]

export const mockDataset = [
  { drug: 'Warfarin', interactions: 42, sideEffects: 18 },
  { drug: 'Ibuprofen', interactions: 31, sideEffects: 12 },
  { drug: 'Simvastatin', interactions: 27, sideEffects: 15 },
  { drug: 'Metformin', interactions: 19, sideEffects: 9 },
  { drug: 'Aspirin', interactions: 34, sideEffects: 14 },
]

export const mockNetwork = {
  nodes: [
    { id: 'Warfarin', group: 1 },
    { id: 'Ibuprofen', group: 2 },
    { id: 'Simvastatin', group: 2 },
    { id: 'Metformin', group: 3 },
    { id: 'Aspirin', group: 3 },
  ],
  links: [
    { source: 'Warfarin', target: 'Ibuprofen', value: 0.87 },
    { source: 'Warfarin', target: 'Aspirin', value: 0.82 },
    { source: 'Ibuprofen', target: 'Aspirin', value: 0.65 },
    { source: 'Simvastatin', target: 'Warfarin', value: 0.44 },
  ],
}

export const mockModels = [
  { name: 'LogReg', auc: 0.89, f1: 0.74 },
  { name: 'RandomForest', auc: 0.91, f1: 0.77 },
  { name: 'XGBoost', auc: 0.93, f1: 0.8 },
]

export const mockArchitecture = [
  'Datasets → Feature engineering',
  'Classical models (LR/RF) + XGBoost',
  'Hybrid fusion and calibration',
  'SHAP explainability',
  'FastAPI + React dashboard',
]
