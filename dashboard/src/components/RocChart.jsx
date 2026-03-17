import { Card, CardHeader, CardBody } from './Primitives'
import styles from './RocChart.module.css'

const Y_LABELS = [
  { label: '0.7', y: 138 },
  { label: '0.8', y: 96  },
  { label: '0.9', y: 55  },
  { label: '1.0', y: 16  },
]

const GRID_Y = [40, 80, 120]

const X_POS = [20, 120, 220, 320, 420, 520, 660]

function lineForScores(scores) {
  const vals = scores.slice(0, X_POS.length).map(s => Math.max(0, Math.min(1, Number(s) || 0)))
  while (vals.length < X_POS.length) {
    vals.push(vals.length ? vals[vals.length - 1] : 0)
  }
  return X_POS.map((x, i) => {
    const y = Math.round(150 - vals[i] * 130)
    return `${x},${y}`
  }).join(' ')
}

export function RocChart({ modelRows = [], result }) {
  const modelScores = modelRows.map(m => Number(m.score) || 0)
  const hybrid = Number(result?.probability || 0)
  const conf = Number(result?.confidence || 0)

  const rocScores = modelScores.length ? [0.65, ...modelScores, hybrid] : [0, 0, 0, 0, 0, 0, 0]
  const prScores = modelScores.length ? [0.55, ...modelScores.map(s => s * conf), hybrid * conf] : [0, 0, 0, 0, 0, 0, 0]

  const ROC_POINTS = lineForScores(rocScores)
  const PR_POINTS = lineForScores(prScores)
  const ROC_FILL = `${ROC_POINTS} 660,155 20,155`
  const PR_FILL = `${PR_POINTS}  660,155 20,155`

  return (
    <Card>
      <CardHeader
        title="Model Performance Monitoring"
        subtitle="Prediction-conditioned model confidence curves"
        right={
          <div className={styles.legend}>
            <span className={styles.legendItem} style={{ '--lc': 'var(--cyan)' }}>
              <span className={styles.legendLine} />
              ROC-AUC
            </span>
            <span className={styles.legendItem} style={{ '--lc': 'var(--amber)' }}>
              <span className={styles.legendLineDashed} />
              PR-AUC
            </span>
          </div>
        }
      />
      <CardBody>
        <div className={styles.chartWrap}>
          <svg viewBox="0 0 700 160" className={styles.svg}>
            <defs>
              <linearGradient id="rocGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#00d4ff" stopOpacity="0.18" />
                <stop offset="100%" stopColor="#00d4ff" stopOpacity="0"    />
              </linearGradient>
              <linearGradient id="prGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#f5a623" stopOpacity="0.12" />
                <stop offset="100%" stopColor="#f5a623" stopOpacity="0"    />
              </linearGradient>
            </defs>

            {GRID_Y.map(y => (
              <line key={y} x1="32" y1={y} x2="685" y2={y}
                stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
            ))}

            {Y_LABELS.map(({ label, y }) => (
              <text key={label} x="28" y={y + 4}
                textAnchor="end" fontSize="9"
                fill="rgba(90,106,138,0.7)"
                fontFamily="'Space Mono', monospace"
              >{label}</text>
            ))}

            <polygon points={ROC_FILL} fill="url(#rocGrad)" />
            <polygon points={PR_FILL}  fill="url(#prGrad)"  />

            <polyline fill="none" stroke="#00d4ff" strokeWidth="2.5"
              strokeLinejoin="round" strokeLinecap="round"
              points={ROC_POINTS}
            />
            <polyline fill="none" stroke="#f5a623" strokeWidth="2"
              strokeLinejoin="round" strokeLinecap="round"
              strokeDasharray="5 3"
              points={PR_POINTS}
            />
          </svg>
        </div>
        {!modelRows.length && <div className={styles.emptyState}>Run prediction to populate chart metrics.</div>}
      </CardBody>
    </Card>
  )
}
