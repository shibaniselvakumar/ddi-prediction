import styles from './Sidebar.module.css'
import { NAV_ITEMS } from '../lib/constants'

export function Sidebar({ active, onNavigate }) {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.logo}>
        <div className={styles.logoMark}>
          <span className={styles.liveDot} />
          Live · NSIDES
        </div>
        <div className={styles.logoTitle}>DDI Intelligence</div>
        <div className={styles.logoSub}>Clinical AI Console v2</div>
      </div>

      <nav className={styles.nav}>
        {NAV_ITEMS.map(item => (
          <button
            key={item.key}
            onClick={() => onNavigate(item.key)}
            className={`${styles.navBtn} ${active === item.key ? styles.navBtnActive : ''}`}
          >
            <span className={styles.glyph}>{item.glyph}</span>
            {item.label}
            {active === item.key && <span className={styles.activePip} />}
          </button>
        ))}
      </nav>

      <footer className={styles.footer}>
        <div className={styles.footerRow}>
          <span className={styles.statusDot} />
          Frontend Online
        </div>
        <div className={styles.footerRow}>
          <span className={styles.statusDot} />
          Inference API Connected
        </div>
        <div className={styles.build}>BUILD 2025.03.17</div>
      </footer>
    </aside>
  )
}
