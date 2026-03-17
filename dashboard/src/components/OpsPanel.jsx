import { Card, CardHeader, CardBody } from './Primitives'
import styles from './OpsPanel.module.css'

export function OpsPanel({ result, modelRows, loading }) {
  const modelCount = modelRows.length
  const shapCount = Array.isArray(result?.shap?.local_top) ? result.shap.local_top.length : 0

  const METRICS = [
    { label: 'Inference Status', value: loading ? 'Running' : (result ? 'Completed' : 'Idle'), accent: 'var(--cyan)' },
    { label: 'Hybrid Probability', value: `${Math.round((Number(result?.probability || 0)) * 100)}%`, accent: 'var(--green)' },
    { label: 'Models Returned', value: String(modelCount), accent: 'var(--amber)' },
    { label: 'Local SHAP Factors', value: String(shapCount), accent: 'var(--cyan)' },
  ]

  const CHECKS = [
    { icon: result ? '✅' : '⚪', text: 'Prediction payload received from API', warn: false },
    { icon: modelCount ? '✅' : '⚪', text: 'Per-model probabilities available', warn: false },
    { icon: shapCount ? '✅' : '⚪', text: 'Local SHAP explainability available', warn: false },
    { icon: result?.agreement ? '✅' : '⚪', text: `Agreement: ${result?.agreement || 'N/A'}`, warn: false },
  ]

  return (
    <div className="grid-2">
      <div className={styles.metricsGrid}>
        {METRICS.map(({ label, value, accent }) => (
          <div key={label} className={styles.metricCard} style={{ '--accent': accent }}>
            <div className={styles.metricLabel}>{label}</div>
            <div className={styles.metricValue}>{value}</div>
          </div>
        ))}
      </div>

      <Card>
        <CardHeader
          title="Governance Checklist"
          subtitle="Pre-deployment readiness"
        />
        <CardBody>
          <div className={styles.checklist}>
            {CHECKS.map(({ icon, text, warn }) => (
              <div
                key={text}
                className={`${styles.checkItem} ${warn ? styles.checkItemWarn : ''}`}
              >
                <span className={styles.checkIcon}>{icon}</span>
                <span>{text}</span>
              </div>
            ))}
          </div>
        </CardBody>
      </Card>
    </div>
  )
}
