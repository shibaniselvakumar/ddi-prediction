import { useState } from 'react'

import '../styles/tokens.css'
import '../styles/global.css'
import '../styles/components.css'

import { useDDIData, usePrediction, usePairSearch } from '../hooks/useDDI'

import { Sidebar }       from '../components/Sidebar'
import { Topbar }        from '../components/Topbar'
import { PredictPanel }  from '../components/PredictPanel'
import { ExplainPanel }  from '../components/ExplainPanel'
import { NetworkPanel }  from '../components/NetworkPanel'
import { RocChart }      from '../components/RocChart'
import { DatasetPanel }  from '../components/DatasetPanel'
import { OpsPanel }      from '../components/OpsPanel'
import { StatStrip, RiskBadge, SectionDivider } from '../components/Primitives'

export default function DashboardPage() {
  const [activeNav, setActiveNav] = useState('predict')

  const { drugs, pairs, loadError }                 = useDDIData()
  const { result, loading, drugA, setDrugA,
    drugB, setDrugB, predict, error,
      shapItems, modelRows, drugOptions, drugBOptions } = usePrediction(drugs, pairs)
  const { search, setSearch, filtered, topSignals } = usePairSearch(pairs)

  const navigate = (key) => {
    setActiveNav(key)
    document.getElementById(`section-${key}`)
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const kpiCells = [
    {
      label:  'Interaction Probability',
      value:  `${Math.round((result?.probability || 0) * 100)}%`,
      helper: 'Hybrid ensemble score',
      accent: 'var(--cyan)',
    },
    {
      label:  'Risk Classification',
      value:  <RiskBadge level={result?.risk || 'Moderate'} />,
      helper: 'Threshold calibrated',
      accent: result?.risk === 'High' ? 'var(--red)' : result?.risk === 'Moderate' ? 'var(--amber)' : 'var(--green)',
    },
    {
      label:  'Confidence Score',
      value:  Number(result?.confidence || 0).toFixed(2),
      helper: 'Ensemble consistency',
      accent: 'var(--amber)',
    },
    {
      label:  'Model Agreement',
      value:  result?.agreement || '0/0',
      helper: 'Voting classifiers',
      accent: 'var(--green)',
    },
  ]

  return (
    <div className="shell">
      <Sidebar active={activeNav} onNavigate={navigate} />

      <main className="main">
        <Topbar onPredict={predict} />

        {error && (
          <div className="card" style={{ marginTop: 12, borderColor: 'rgba(255,94,125,0.4)' }}>
            <div className="card__body" style={{ color: 'var(--red)' }}>Prediction error: {error}</div>
          </div>
        )}

        {result?.syntheticPair && (
          <div className="card" style={{ marginTop: 12, borderColor: 'rgba(245,166,35,0.4)' }}>
            <div className="card__body" style={{ color: 'var(--amber)' }}>
              Pair not found in cached pair-features. Prediction used synthesized pair features from per-drug statistics.
            </div>
          </div>
        )}

        {loadError && (
          <div className="card" style={{ marginTop: 12, borderColor: 'rgba(245,166,35,0.4)' }}>
            <div className="card__body" style={{ color: 'var(--amber)' }}>Data load warning: {loadError}</div>
          </div>
        )}

        <StatStrip cells={kpiCells} />

        {/* ── Predict ── */}
        <div id="section-predict">
          <SectionDivider label="Risk Prediction" />
          <PredictPanel
            drugs={drugOptions}
            drugBOptions={drugBOptions}
            drugA={drugA}   setDrugA={setDrugA}
            drugB={drugB}   setDrugB={setDrugB}
            onPredict={predict}
            loading={loading}
            result={result}
          />
        </div>

        {/* ── Explainability ── */}
        <div id="section-explain">
          <SectionDivider label="Model Explainability" />
          <ExplainPanel shapItems={shapItems} modelRows={modelRows} />
        </div>

        {/* ── Network + ROC ── */}
        <div id="section-network">
          <SectionDivider label="Signal Network" />
          <NetworkPanel topSignals={topSignals} result={result} />
          <div style={{ marginTop: 16 }}>
            <RocChart modelRows={modelRows} result={result} />
          </div>
        </div>

        {/* ── Dataset ── */}
        <div id="section-dataset">
          <SectionDivider label="Data Explorer" />
          <DatasetPanel
            drugs={drugs}
            pairs={pairs}
            filtered={filtered}
            search={search}
            setSearch={setSearch}
          />
        </div>

        {/* ── Operations ── */}
        <div id="section-ops" style={{ marginBottom: 48 }}>
          <SectionDivider label="Operations" />
          <OpsPanel result={result} modelRows={modelRows} loading={loading} />
        </div>
      </main>
    </div>
  )
}
