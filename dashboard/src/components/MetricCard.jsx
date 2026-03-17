export default function MetricCard({ label, value, helper, tone = 'default' }) {
  const toneClass = {
    default: 'border-slate-200 bg-white',
    positive: 'border-emerald-200 bg-emerald-50',
    warning: 'border-amber-200 bg-amber-50',
    danger: 'border-rose-200 bg-rose-50',
  }[tone]

  return (
    <div className={`relative overflow-hidden rounded-2xl border px-4 py-4 card-shadow ${toneClass}`}>
      <p className="text-xs uppercase tracking-[0.2em] text-slate-600">{label}</p>
      <p className="mt-2 font-mono text-3xl font-semibold text-slate-900" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
        {value}
      </p>
      {helper && <p className="mt-2 text-xs text-slate-600">{helper}</p>}
    </div>
  )
}
