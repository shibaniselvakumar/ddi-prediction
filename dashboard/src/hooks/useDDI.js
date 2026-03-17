import { useEffect, useMemo, useState } from 'react'
import { fetchDatasetPairs, fetchDrugs, predictInteraction } from '../api/client'

/**
 * Loads drug list and dataset pairs on mount.
 * Falls back to static data if the API is unavailable.
 */
export function useDDIData() {
  const [drugs, setDrugs]   = useState([])
  const [pairs, setPairs]   = useState([])
  const [ready, setReady]   = useState(false)
  const [loadError, setLoadError] = useState('')

  useEffect(() => {
    let cancelled = false

    fetchDrugs()
      .then((items) => {
        if (cancelled) return
        if (Array.isArray(items) && items.length) setDrugs(items)
      })
      .catch((err) => {
        if (cancelled) return
        setLoadError(err?.message || 'Failed to load drugs')
      })

    // Fast initial paint with sample, then replace with full dataset.
    fetchDatasetPairs(1000)
      .then((rows) => {
        if (cancelled) return
        if (Array.isArray(rows) && rows.length) {
          setPairs(rows)

          // Fallback: if /drugs endpoint is empty or delayed, derive options from pairs.
          setDrugs((prev) => {
            if (prev.length) return prev
            const map = new Map()
            rows.forEach((r) => {
              if (r?.drug1_name && r?.drug_1_rxnorm_id) map.set(String(r.drug1_name), String(r.drug_1_rxnorm_id))
              if (r?.drug2_name && r?.drug_2_rxnorm_id) map.set(String(r.drug2_name), String(r.drug_2_rxnorm_id))
            })
            return [...map.entries()].map(([name, id]) => ({ id, name }))
          })
        }
      })
      .catch((err) => {
        if (cancelled) return
        setLoadError(err?.message || 'Failed to load dataset sample')
      })
      .finally(() => {
        if (!cancelled) setReady(true)
      })

    fetchDatasetPairs()
      .then((rows) => {
        if (cancelled) return
        if (Array.isArray(rows) && rows.length) setPairs(rows)
      })
      .catch(() => {
        // Keep sampled pairs if full load fails/timeouts.
      })

    return () => {
      cancelled = true
    }
  }, [])

  return { drugs, pairs, ready, loadError }
}

/**
 * Manages the prediction state machine.
 * Returns current result, loading flag, and a trigger function.
 */
export function usePrediction(drugs, pairs) {
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')
  const [drugA,   setDrugA]   = useState('')
  const [drugB,   setDrugB]   = useState('')

  const drugOptions = useMemo(() => drugs, [drugs])
  const drugBOptions = useMemo(() => drugs, [drugs])

  // Sync selectors when drug list or dataset pairs load
  useEffect(() => {
    if (pairs.length) {
      const first = pairs[0]
      const candidateA = first?.drug1_name
      const candidateB = first?.drug2_name
      const hasA = drugOptions.some(d => d.name === candidateA)
      const hasB = drugOptions.some(d => d.name === candidateB)
      if (hasA && hasB) {
        setDrugA(candidateA)
        setDrugB(candidateB)
        return
      }
    }
    if (drugOptions.length >= 2) {
      setDrugA(drugOptions[0].name)
      setDrugB(drugOptions[1].name)
    }
  }, [drugOptions, pairs])

  useEffect(() => {
    if (!drugA || !drugBOptions.length) return
    if (!drugBOptions.some((d) => d.name === drugB)) {
      setDrugB(drugBOptions[0].name)
    }
  }, [drugA, drugB, drugBOptions])

  const predict = () => {
    const a = drugs.find(d => d.name === drugA)
    const b = drugs.find(d => d.name === drugB)
    if (!a?.id || !b?.id) {
      setError('Drug list not loaded yet. Please wait and try again.')
      return
    }

    setLoading(true)
    setError('')
    predictInteraction(a.id, b.id)
      .then(res => setResult({
        ...res,
        drugAName: res.drugAName || a.name,
        drugBName: res.drugBName || b.name,
      }))
      .catch(err => {
        setResult(null)
        setError(err?.message || 'Prediction failed')
      })
      .finally(() => setLoading(false))
  }

  const shapItems = useMemo(() => {
    const local = result?.shap?.local_top
    return (Array.isArray(local) && local.length) ? local.slice(0, 6) : []
  }, [result])

  const modelRows = useMemo(() => {
    const src = result?.perModel || {}
    const entries = Object.entries(src)
    return entries.length
      ? entries.map(([key, score]) => ({
          name:  key.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase()),
          score: Number(score) || 0,
        }))
      : []
  }, [result])

  return {
    result,
    loading,
    error,
    drugA,
    setDrugA,
    drugB,
    setDrugB,
    predict,
    shapItems,
    modelRows,
    drugOptions,
    drugBOptions,
  }
}

/**
 * Filters and sorts pairs for the dataset explorer.
 */
export function usePairSearch(pairs) {
  const [search, setSearch] = useState('')

  const filtered = useMemo(() => {
    if (!search.trim()) return pairs
    const q = search.toLowerCase()
    return pairs.filter(p =>
      `${p.drug1_name ?? ''} ${p.drug2_name ?? ''}`.toLowerCase().includes(q)
    )
  }, [pairs, search])

  const topSignals = useMemo(() =>
    [...pairs]
      .sort((a, b) => Number(b.pair_max_prr ?? 0) - Number(a.pair_max_prr ?? 0))
      .slice(0, 6),
    [pairs]
  )

  return { search, setSearch, filtered, topSignals }
}
