import daisyui from "daisyui";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: {} },
  plugins: [daisyui],
  daisyui: {
    themes: [
      "forest",   // dark mode  — primary: emerald, bg: dark green
      {
        raglight: {
          "primary":          "#059669",   // emerald-600  — same green family
          "primary-content":  "#ffffff",
          "secondary":        "#065f46",   // emerald-800
          "accent":           "#00FF9D",   // neon green accent (kept from dark)
          "neutral":          "#1f2937",
          "base-100":         "#ffffff",   // card backgrounds
          "base-200":         "#f0fdf4",   // page background — light green tint
          "base-300":         "#dcfce7",   // header/sidebar
          "base-content":     "#064e3b",   // dark green text
          "info":             "#0ea5e9",
          "success":          "#059669",
          "warning":          "#f59e0b",
          "error":            "#dc2626",
        },
      },
    ],
  },
};