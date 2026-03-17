import { Card, CardHeader, CardBody } from './Primitives'
import styles from './NetworkPanel.module.css'

const POSITIONS = [
  { x: 100, y: 48, r: 22 },
  { x: 420, y: 52, r: 22 },
  { x: 155, y: 198, r: 22 },
  { x: 388, y: 188, r: 22 },
  { x: 72, y: 150, r: 18 },
  { x: 448, y: 135, r: 18 },
]

export function NetworkPanel({ topSignals, result }) {
  const predictionNodes = [result?.drugAName, result?.drugBName].filter(Boolean)
  const rankedNodes = topSignals
    .flatMap(row => [row.drug1_name, row.drug2_name])
    .filter(Boolean)

  const labels = [...new Set([...predictionNodes, ...rankedNodes])].slice(0, 6)
  const nodes = labels.map((label, idx) => ({
    id: `${label}-${idx}`,
    label,
    ...POSITIONS[idx],
  }))
  const edges = nodes.map(n => ({ x1: 260, y1: 120, x2: n.x, y2: n.y }))

  return (
    <div className="grid-60-40">
      <Card>
        <CardHeader
          title="Interaction Network"
          subtitle="Topology of high-risk drug candidates"
        />
        <CardBody>
          <div className={styles.networkWrap}>
            <svg viewBox="0 0 520 250" className={styles.svg}>
              <defs>
                <radialGradient id="hubGrad" cx="50%" cy="50%" r="50%">
                  <stop offset="0%"   stopColor="#00d4ff" stopOpacity="0.35" />
                  <stop offset="100%" stopColor="#1e3a8a" stopOpacity="0.1"  />
                </radialGradient>
                <radialGradient id="nodeGrad" cx="50%" cy="50%" r="50%">
                  <stop offset="0%"   stopColor="#1e3a8a" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#0d1424" stopOpacity="0.9" />
                </radialGradient>
                <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>

              {edges.map((e, i) => (
                <line
                  key={i}
                  x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
                  stroke="rgba(0,212,255,0.2)"
                  strokeWidth="1.5"
                  strokeDasharray="5 4"
                />
              ))}

              {nodes.map(n => (
                <g key={n.id}>
                  <circle cx={n.x} cy={n.y} r={n.r}
                    fill="url(#nodeGrad)"
                    stroke="rgba(0,212,255,0.3)"
                    strokeWidth="1"
                    filter="url(#glow)"
                  />
                  <text
                    x={n.x} y={n.y}
                    textAnchor="middle" dominantBaseline="middle"
                    fontSize="8" fill="#8aabcc"
                    fontFamily="'Space Mono', monospace"
                  >
                    {n.label}
                  </text>
                </g>
              ))}

              {/* Hub */}
              <circle cx="260" cy="120" r="30"
                fill="url(#hubGrad)"
                stroke="#00d4ff" strokeWidth="1.5"
                filter="url(#glow)"
              />
              <text x="260" y="115" textAnchor="middle" fontSize="9"
                fill="#00d4ff" fontFamily="'Space Mono', monospace" fontWeight="700">DDI</text>
              <text x="260" y="127" textAnchor="middle" fontSize="9"
                fill="#00d4ff" fontFamily="'Space Mono', monospace" fontWeight="700">HUB</text>
              <circle cx="260" cy="120" r="4" fill="#00d4ff" />
            </svg>
            {!nodes.length && <div className={styles.emptyNetwork}>Run prediction to build network context.</div>}
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="High-Signal Pairs"
          subtitle="Top PRR entries in current dataset"
        />
        <CardBody style={{ paddingTop: 12 }}>
          {topSignals.length ? (
            topSignals.map((row, i) => (
              <div key={i} className={styles.signalRow}>
                <div>
                  <div className={styles.signalDrugA}>{row.drug1_name}</div>
                  <div className={styles.signalDrugB}>+ {row.drug2_name}</div>
                </div>
                <div className={styles.signalPrr}>
                  <span className={styles.prrValue}>{Number(row.pair_max_prr ?? 0).toFixed(2)}</span>
                  <span className={styles.prrLabel}>MAX PRR</span>
                </div>
              </div>
            ))
          ) : (
            <div className={styles.emptySignals}>No dataset rows available.</div>
          )}
        </CardBody>
      </Card>
    </div>
  )
}
