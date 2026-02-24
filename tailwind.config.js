/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                accent: 'var(--accent-red)',
                main: 'var(--text-main)',
                dim: 'var(--text-dim)',
                panel: 'var(--panel-bg)',
                'panel-bg-solid': 'var(--panel-bg-solid)',
            },
            fontFamily: {
                dot: ['DotGothic16', 'sans-serif'],
                mono: ['JetBrains Mono', 'monospace'],
            }
        }
    },
    plugins: [],
}
