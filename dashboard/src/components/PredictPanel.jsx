import { Card, CardHeader, CardBody, Shimmer } from './Primitives'
import styles from './PredictPanel.module.css'

function RiskGauge({ marker }) {
  return (
    <div className={styles.gaugeSection}>
      <div className={styles.gaugeHeader}>
        <span className={styles.gaugeLabel}>Risk Gauge</span>
        <span className={styles.gaugeValue}>{marker}%</span>
      </div>
      <div className={styles.gaugeTrack}>
        <div className={styles.gaugeThumb} style={{ left: `${marker}%` }} />
      </div>
      <div className={styles.gaugeMarks}>
        <span>LOW</span>
        <span>MODERATE</span>
        <span>HIGH</span>
      </div>
    </div>
  )
}

function ClinicalSummary({ result, drugA, drugB }) {
  return (
    <div className={styles.summary}>
      <p className={styles.summaryEyebrow}>Clinical Summary</p>
      <p className={styles.summaryText}>
        Predicted interaction for{' '}
        <strong>{result.drugAName || drugA}</strong>
        {' '}+{' '}
        <strong>{result.drugBName || drugB}</strong>
        {' '}indicates{' '}
        <strong>{(result.risk || 'Moderate').toLowerCase()} risk</strong>
        {' '}with confidence{' '}
        <strong>{Number(result.confidence || 0).toFixed(2)}</strong>.
        Primary signal contributors include pharmacovigilance disproportionality
        and shared adverse effect pathways.
      </p>
    </div>
  )
}

export function PredictPanel({ drugs, drugBOptions = [], drugA, setDrugA, drugB, setDrugB, onPredict, loading, result }) {
  const marker = Math.max(0, Math.min(Math.round((result?.probability || 0) * 100), 100))
  const hasDrugs = drugs.length > 1
  const hasDrugBOptions = drugBOptions.length > 0

  return (
    <Card>
      <CardHeader
        title="Clinical Risk Prediction"
        subtitle="Select a drug pair and run explainable real-time inference"
        right={
          <button className="btn btn--primary" onClick={onPredict} disabled={loading || !hasDrugs}>
            {loading ? 'Computing…' : 'Predict Interaction'}
          </button>
        }
      />
      <CardBody>
        <div className={styles.selectors}>
          <div className={styles.selectorGroup}>
            <label className={styles.selectorLabel}>Drug A</label>
            <select value={drugA} onChange={e => setDrugA(e.target.value)} className="input" disabled={!hasDrugs}>
              {!hasDrugs && <option value="">Loading drugs...</option>}
              {drugs.map(d => <option key={d.id} value={d.name}>{d.name}</option>)}
            </select>
          </div>

          <div className={styles.vsBadge}>VS</div>

          <div className={styles.selectorGroup}>
            <label className={styles.selectorLabel}>Drug B</label>
            <select value={drugB} onChange={e => setDrugB(e.target.value)} className="input" disabled={!hasDrugs || !hasDrugBOptions}>
              {!hasDrugs && <option value="">Loading drugs...</option>}
              {hasDrugs && !hasDrugBOptions && <option value="">No compatible pairs</option>}
              {drugBOptions.map(d => <option key={d.id} value={d.name}>{d.name}</option>)}
            </select>
          </div>
        </div>

        {!hasDrugs && <div className={styles.summaryText}>Waiting for API drug list...</div>}

        {loading && <Shimmer />}

        <RiskGauge marker={marker} />
        <ClinicalSummary result={result || {}} drugA={drugA} drugB={drugB} />
      </CardBody>
    </Card>
  )
}
