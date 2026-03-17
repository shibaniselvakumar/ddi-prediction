import styles from './Topbar.module.css'

export function Topbar({ onPredict }) {
  return (
    <header className={styles.topbar}>
      <div className={styles.left}>
        <p className={styles.eyebrow}>◈ Predictive Safety Monitoring</p>
        <h1 className={styles.headline}>
          Drug–Drug<br />
          <span className={styles.accent}>Interaction</span> Console
        </h1>
        <p className={styles.description}>
          Explainable AI for clinical interaction risk scoring, SHAP attribution,
          and pharmacovigilance signal analysis.
        </p>
      </div>

      <div className={styles.right}>
        <span className="chip">NSIDES Dataset</span>
        <span className="chip">Inference Mode</span>
        <button className="btn btn--ghost" onClick={onPredict}>Run Prediction</button>
      </div>
    </header>
  )
}
