import { Card, CardHeader, CardBody } from './Primitives'
import styles from './ExplainPanel.module.css'

function ShapRow({ feature, shap }) {
  const width = Math.min(Math.abs(shap) * 320, 100)
  return (
    <div className={styles.shapRow}>
      <span className={styles.shapFeature}>{feature}</span>
      <div className={styles.shapTrack}>
        <div className={styles.shapFill} style={{ width: `${width}%` }} />
      </div>
      <span className={styles.shapVal}>+{Number(shap).toFixed(3)}</span>
    </div>
  )
}

function ModelRow({ name, score }) {
  const pct = Math.round(score * 100)
  const label = score > 0 && score < 0.001 ? '<0.001' : score.toFixed(3)
  return (
    <div className={styles.modelRow}>
      <div className={styles.modelHeader}>
        <span className={styles.modelName}>{name}</span>
        <span className={styles.modelScore}>{label}</span>
      </div>
      <div className={styles.modelTrack}>
        <div className={styles.modelFill} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export function ExplainPanel({ shapItems, modelRows }) {
  return (
    <div className="grid-2">
      <Card>
        <CardHeader
          title="SHAP Feature Attribution"
          subtitle="Top local factors driving the current inference"
        />
        <CardBody>
          {shapItems.length ? (
            shapItems.map(item => (
              <ShapRow key={item.feature} feature={item.feature} shap={item.shap} />
            ))
          ) : (
            <div className={styles.emptyState}>Run prediction to generate SHAP attributions.</div>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="Per-Model Probability"
          subtitle="Classifier-level confidence scores"
        />
        <CardBody>
          {modelRows.length ? (
            modelRows.map(m => (
              <ModelRow key={m.name} name={m.name} score={m.score} />
            ))
          ) : (
            <div className={styles.emptyState}>Run prediction to view per-model outputs.</div>
          )}
        </CardBody>
      </Card>
    </div>
  )
}
