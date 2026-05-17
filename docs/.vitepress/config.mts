import { defineConfig } from "vitepress";

export default defineConfig({
  title: "FDM-EDL",
  description:
    "Finite-difference electrical double layer simulations with JAX and unit-aware inputs",
  base: "/fdm-edl/",
  srcExclude: ["agents/**", "CHANGELOG.md"],
  ignoreDeadLinks: [/^(\.\.?\/)?src\/fdm_edl\/.*:\d+$/, "./LICENSE"],
  themeConfig: {
    nav: [
      { text: "Home", link: "/" },
      { text: "Docs", link: "/docs/" },
      { text: "API Reference", link: "/api/fdm_edl.html", target: "_self" },
      { text: "GitHub", link: "https://github.com/ChiahsinChu/fdm-edl" },
    ],
    sidebar: {
      "/docs/": [
        {
          text: "Documentation",
          items: [
            { text: "Getting Started", link: "/docs/" },
            { text: "Examples", link: "/docs/examples/" },
            { text: "Parameters", link: "/docs/config_params.md" },
            { text: "Development", link: "/docs/development/" },
          ],
        },
      ],
    },
  },
});
