import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'

export default function NetworkGraph({ data, highlightPair }) {
  const svgRef = useRef(null)
  const [nodes, setNodes] = useState([])
  const [links, setLinks] = useState([])

  useEffect(() => {
    const simNodes = data.nodes.map((d) => ({ ...d }))
    const simLinks = data.links.map((l) => ({ ...l }))

    const simulation = d3
      .forceSimulation(simNodes)
      .force('link', d3.forceLink(simLinks).id((d) => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-250))
      .force('center', d3.forceCenter(320 / 2, 260 / 2))
      .stop()

    simulation.tick(80)
    setNodes(simNodes)
    setLinks(simLinks)
  }, [data])

  return (
    <svg ref={svgRef} viewBox="0 0 320 260" className="w-full rounded-xl border border-slate-200 bg-slate-50">
      <defs>
        <linearGradient id="linkGradient" x1="0%" x2="100%">
          <stop offset="0%" stopColor="#4754f6" />
          <stop offset="100%" stopColor="#10b981" />
        </linearGradient>
      </defs>
      {links.map((link, idx) => {
        const isHighlight =
          highlightPair &&
          ((link.source.id === highlightPair[0] && link.target.id === highlightPair[1]) ||
            (link.source.id === highlightPair[1] && link.target.id === highlightPair[0]))
        return (
          <line
            key={idx}
            x1={link.source.x}
            y1={link.source.y}
            x2={link.target.x}
            y2={link.target.y}
            stroke={isHighlight ? '#f59e0b' : 'url(#linkGradient)'}
            strokeWidth={isHighlight ? 3 : 2}
            opacity={0.85}
            strokeLinecap="round"
          />
        )
      })}
      {nodes.map((node) => (
        <g key={node.id}>
          <circle
            cx={node.x}
            cy={node.y}
            r={highlightPair?.includes(node.id) ? 12 : 10}
            fill={highlightPair?.includes(node.id) ? '#f59e0b' : '#4754f6'}
            opacity={0.9}
          />
          <text
            x={node.x}
            y={node.y - 16}
            textAnchor="middle"
            fontSize="10"
            fill="#e2e8f0"
            fontWeight="600"
          >
            {node.id}
          </text>
        </g>
      ))}
    </svg>
  )
}
