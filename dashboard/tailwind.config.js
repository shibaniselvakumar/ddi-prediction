/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        ink: {
          50: '#f7f8fb',
          100: '#eef0f6',
          200: '#dde0ec',
          300: '#c3c7d7',
          400: '#9ea4bb',
          500: '#7c84a3',
          600: '#5f668a',
          700: '#4b5170',
          800: '#3e435c',
          900: '#303349',
        },
        accent: {
          50: '#f5f6ff',
          100: '#e6e9ff',
          200: '#c7ceff',
          300: '#9ca8ff',
          400: '#6a79ff',
          500: '#4754f6',
          600: '#313ae0',
          700: '#262db3',
          800: '#20288f',
          900: '#1d276f',
        },
        risk: {
          low: '#10b981',
          medium: '#f59e0b',
          high: '#ef4444',
        },
      },
      boxShadow: {
        glass: '0 20px 70px rgba(27, 39, 94, 0.20)',
      },
      backgroundImage: {
        grid: 'radial-gradient(circle at 1px 1px, rgba(255,255,255,0.12) 1px, transparent 0)',
        beam: 'linear-gradient(135deg, rgba(71,84,246,0.12), rgba(16,185,129,0.08), rgba(96,165,250,0.10))',
      },
    },
  },
  plugins: [],
}
