/* =========================
   📁 components/PredictionPanel.jsx
========================= */
export default function PredictionPanel({
  drugA = '',
  setDrugA = () => {},
  drugB = '',
  setDrugB = () => {},
  onPredict = () => {},
  loading = false,
  result = null,
}) {
  const risk = result?.risk || 'N/A'
  const probabilityPct = Math.round((Number(result?.probability || 0)) * 100)
  const confidencePct = Math.round((Number(result?.confidence || 0)) * 100)

  return (
    <div className="grid lg:grid-cols-2 gap-6">
      <div className="bg-white p-6 rounded-2xl shadow">
        <h2 className="font-semibold text-lg mb-4">Select Drugs</h2>

        <input
          className="w-full border p-3 rounded-lg mb-3"
          placeholder="Drug A"
          value={drugA}
          onChange={(e) => setDrugA(e.target.value)}
        />
        <input
          className="w-full border p-3 rounded-lg mb-3"
          placeholder="Drug B"
          value={drugB}
          onChange={(e) => setDrugB(e.target.value)}
        />

        <button className="w-full bg-blue-600 text-white py-3 rounded-lg" onClick={onPredict} disabled={loading}>
          {loading ? 'Predicting…' : 'Predict Interaction'}
        </button>
      </div>

      <div className="bg-white p-6 rounded-2xl shadow">
        <h2 className="font-semibold text-lg mb-4">Result</h2>

        <div className="space-y-4">
          <div>
            <p className="text-sm text-slate-500">Risk</p>
            <p className="text-xl font-bold text-red-600">{risk}</p>
          </div>

          <div>
            <p className="text-sm text-slate-500">Probability</p>
            <div className="w-full bg-slate-200 h-2 rounded">
              <div className="h-2 bg-red-500 rounded" style={{ width: `${probabilityPct}%` }} />
            </div>
          </div>

          <div>
            <p className="text-sm text-slate-500">Confidence</p>
            <div className="w-full bg-slate-200 h-2 rounded">
              <div className="h-2 bg-blue-500 rounded" style={{ width: `${confidencePct}%` }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}