/* =========================
   📁 components/Header.jsx
========================= */
export default function Header() {
  return (
    <div className="bg-white border-b px-6 py-4 flex justify-between items-center">
      <div>
        <h1 className="text-2xl font-semibold">Drug Interaction Intelligence</h1>
        <p className="text-sm text-slate-500">
          AI-powered clinical decision support
        </p>
      </div>

      <button className="bg-blue-600 text-white px-4 py-2 rounded-xl shadow">
        Run Prediction
      </button>
    </div>
  );
}