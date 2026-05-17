import { defineConfig } from "vitepress";

export default defineConfig({
  title: "FDM-EDL",
  description: "Finite-difference electrical double layer simulations with JAX and unit-aware inputs",
  srcExclude: ["agents/**", "CHANGELOG.md"],
  ignoreDeadLinks: [
    /^(\.\.?\/)?src\/fdm_edl\/.*:\d+$/,
    "./LICENSE",
  ],
  themeConfig: {
    nav: [
      { text: "Home", link: "/" },
      { text: "API Reference", link: "/reference/" },
      { text: "GitHub", link: "https://github.com/ChiahsinChu/fdm-edl" },
    ],
    // sidebar: {
    //   "/reference/": [
    //     {
    //       text: "API Reference",
    //       items: [
    //         { text: "Overview", link: "/reference/" },
    //         { text: "fdm_edl", link: "/reference/fdm_edl.html" },
    //       ],
    //     },
    //   ],
    // },
  },
});
