import { useEffect, useMemo, useState } from 'react'
import { Card, CardHeader, CardBody } from './Primitives'
import styles from './DatasetPanel.module.css'

function DatasetStats({ drugsCount, pairsCount, avgPrr }) {
  const stats = [
    { label: 'Unique Drugs',   value: drugsCount, accent: 'var(--cyan)'  },
    { label: 'Loaded Pairs',   value: pairsCount, accent: 'var(--amber)' },
    { label: 'Avg PRR Signal', value: avgPrr,     accent: 'var(--green)' },
  ]

  return (
    <div className={styles.statsRow}>
      {stats.map(({ label, value, accent }) => (
        <div key={label} className={styles.statBox} style={{ '--accent': accent }}>
          <div className={styles.statLabel}>{label}</div>
          <div className={styles.statValue}>{value}</div>
        </div>
      ))}
    </div>
  )
}

function PrrPill({ value }) {
  return <span className={styles.prrPill}>{value}</span>
}

export function DatasetPanel({ drugs, pairs, filtered, search, setSearch }) {
  const PAGE_SIZE = 10
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE)

  useEffect(() => {
    setVisibleCount(PAGE_SIZE)
  }, [search, filtered.length])

  const avgPrr = useMemo(() => (
    pairs.length
      ? (pairs.reduce((acc, row) => acc + Number(row.pair_avg_prr ?? 0), 0) / pairs.length).toFixed(3)
      : '0.000'
  ), [pairs])

  const visibleRows = useMemo(() => filtered.slice(0, visibleCount), [filtered, visibleCount])
  const hasMore = visibleCount < filtered.length

  return (
    <div>
      <DatasetStats drugsCount={drugs.length} pairsCount={pairs.length} avgPrr={avgPrr} />

      <Card>
        <CardHeader
          title="Pharmacovigilance Pairs"
          subtitle="NSIDES dataset — disproportionality signals"
          right={
            <div className={styles.tableControls}>
              <input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search drug name…"
                className={`input ${styles.searchInput}`}
              />
              <span className={styles.rowCount}>showing {visibleRows.length} / {filtered.length}</span>
            </div>
          }
        />
        <CardBody style={{ padding: 0 }}>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Drug A</th>
                  <th>Drug B</th>
                  <th>Max PRR</th>
                  <th>Avg PRR</th>
                </tr>
              </thead>
              <tbody>
                {visibleRows.length ? (
                  visibleRows.map((row, i) => (
                    <tr key={i}>
                      <td className={styles.drugA}>{row.drug1_name || '—'}</td>
                      <td className={styles.drugB}>{row.drug2_name || '—'}</td>
                      <td><PrrPill value={Number(row.pair_max_prr ?? 0).toFixed(3)} /></td>
                      <td className={styles.avgPrr}>{Number(row.pair_avg_prr ?? 0).toFixed(3)}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={4} style={{ textAlign: 'center', padding: '18px 10px' }}>No rows to display.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          {hasMore && (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '12px' }}>
              <button className="btn btn--ghost" onClick={() => setVisibleCount(filtered.length)}>
                Load all rows
              </button>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  )
}
