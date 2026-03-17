/* =========================
   📁 components/SystemPanel.jsx
========================= */
export default function SystemPanel() {
  return (
    <div className="bg-white p-6 rounded-2xl shadow">
      <h2 className="text-lg font-semibold mb-4">System Status</h2>

      <ul className="space-y-2 text-sm">
        <li>✅ API Connected</li>
        <li>⚡ Latency: 118ms</li>
        <li>📈 Uptime: 99.9%</li>
      </ul>
    </div>
  );
}