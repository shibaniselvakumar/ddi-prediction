/* =========================
   📁 components/ExplainabilityPanel.jsx
========================= */
export default function ExplainabilityPanel({ features = [] }) {

  return (
    <div className="bg-white p-6 rounded-2xl shadow">
      <h2 className="text-lg font-semibold mb-4">Explainability</h2>

      {features.length ? (
        features.map((f) => (
          <div key={f.name} className="mb-3">
            <div className="flex justify-between text-sm">
              <span>{f.name}</span>
              <span>{f.val}%</span>
            </div>
            <div className="bg-slate-200 h-2 rounded">
              <div className="bg-green-500 h-2 rounded" style={{ width: `${f.val}%` }} />
            </div>
          </div>
        ))
      ) : (
        <p className="text-sm text-slate-500">Run prediction to view feature contributions.</p>
      )}
    </div>
  )
}