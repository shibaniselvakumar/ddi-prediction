export default function SectionCard({ title, subtitle, children, actions }) {
  return (
    <div className="glass relative overflow-hidden rounded-2xl border border-slate-200 p-5 card-shadow">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-slate-900" style={{ fontFamily: 'Poppins, Inter, sans-serif' }}>
            {title}
          </h3>
          {subtitle && <p className="text-sm text-slate-600">{subtitle}</p>}
        </div>
        {actions}
      </div>
      {children}
    </div>
  )
}
