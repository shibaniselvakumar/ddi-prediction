import axios from 'axios'

const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8011'

const api = axios.create({
  baseURL: apiBase,
  timeout: 60000,
})

export const fetchDrugs = async () => {
  try {
    const { data } = await api.get('/drugs')
    if (Array.isArray(data)) return data
    return data?.drugs || []
  } catch (err) {
    throw new Error(err?.response?.data?.detail || err?.message || 'Failed to load drugs')
  }
}

export const predictInteraction = async (drugAId, drugBId) => {
  try {
    const { data } = await api.post('/predict', null, {
      params: { drug_a_id: drugAId, drug_b_id: drugBId },
    })

    return {
      drugAId: data?.drug_a_id,
      drugBId: data?.drug_b_id,
      drugAName: data?.drug_a_name,
      drugBName: data?.drug_b_name,
      probability: Number(data?.hybrid_prob ?? 0),
      risk: String(data?.risk || 'Moderate'),
      confidence: Number(data?.confidence ?? 0),
      agreement: String(data?.agreement || '0/0'),
      perModel: data?.proba || {},
      syntheticPair: Boolean(data?.synthetic_pair),
      shap: data?.shap || { global_top: [], local_top: [] },
      features: data?.features || {},
    }
  } catch (err) {
    throw new Error(err?.response?.data?.detail || err?.message || 'Prediction failed')
  }
}

export const fetchDatasetPairs = async (sample = null) => {
  try {
    const params = Number.isFinite(sample) ? { sample } : undefined
    const timeout = Number.isFinite(sample) ? 30000 : 120000
    const { data } = await api.get('/dataset/pairs', { params, timeout })
    return Array.isArray(data) ? data : []
  } catch (err) {
    throw new Error(err?.response?.data?.detail || err?.message || 'Failed to load dataset pairs')
  }
}

export default api
