import { getRiskMeta } from '../lib/constants'

// ── Badge ────────────────────────────────────────────────────────────────────

export function RiskBadge({ level, animated = true }) {
  const meta = getRiskMeta(level)
  return (
    <span className={`badge ${meta.cls} ${animated ? 'badge--animated' : ''}`}>
      <span className="badge__dot" />
      {meta.label}
    </span>
  )
}

// ── Card ─────────────────────────────────────────────────────────────────────

export function Card({ children, accentColor, className = '', style = {} }) {
  return (
    <div
      className={`card ${accentColor ? 'card--accent-top' : ''} ${className}`}
      style={{ '--accent': accentColor, ...style }}
    >
      {children}
    </div>
  )
}

export function CardHeader({ title, subtitle, right }) {
  return (
    <div className="card__header">
      <div>
        <div className="card__title">{title}</div>
        {subtitle && <div className="card__subtitle">{subtitle}</div>}
      </div>
      {right}
    </div>
  )
}

export function CardBody({ children, style = {} }) {
  return <div className="card__body" style={style}>{children}</div>
}

// ── Section divider ───────────────────────────────────────────────────────────

export function SectionDivider({ label }) {
  return (
    <div className="section-divider">
      <span className="section-divider__label">{label}</span>
      <div className="section-divider__line" />
    </div>
  )
}

// ── Stat strip ────────────────────────────────────────────────────────────────

export function StatStrip({ cells }) {
  return (
    <div className="stat-strip">
      {cells.map(({ label, value, helper, accent }, i) => (
        <div
          key={label}
          className="stat-cell"
          style={{ '--accent': accent, animationDelay: `${i * 60}ms` }}
        >
          <div className="stat-cell__label">{label}</div>
          <div className={`stat-cell__value ${typeof value !== 'string' ? 'stat-cell__value--sm' : ''}`}
               style={{ marginTop: typeof value !== 'string' ? 6 : 0 }}>
            {value}
          </div>
          {helper && <div className="stat-cell__helper">{helper}</div>}
        </div>
      ))}
    </div>
  )
}

// ── Loading shimmer ───────────────────────────────────────────────────────────

export function Shimmer() {
  return <div className="shimmer" />
}

// ── Eyebrow label ─────────────────────────────────────────────────────────────

export function Eyebrow({ children, style = {} }) {
  return <p className="eyebrow" style={style}>{children}</p>
}
