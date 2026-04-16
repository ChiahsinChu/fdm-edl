# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import mkdocs_gen_files

PACKAGE = "fdm_edl"  # change this
SRC_DIRS = [Path("src"), Path(".")]  # supports both src-layout and flat-layout


def find_package_dir():
    for base in SRC_DIRS:
        pkg = base / PACKAGE
        if pkg.is_dir():
            return pkg
    raise SystemExit(
        f"Can't find package dir for {PACKAGE}. Set PACKAGE/SRC_DIRS correctly."
    )


pkg_dir = find_package_dir()

nav = mkdocs_gen_files.Nav()

for path in sorted(pkg_dir.rglob("*.py")):
    if path.name == "__init__.py":
        continue

    # Convert file path to python module path: your_package/foo/bar.py -> your_package.foo.bar
    rel = path.relative_to(pkg_dir).with_suffix("")
    module = ".".join([PACKAGE, *rel.parts])

    doc_path = Path("reference") / rel.with_suffix(
        ".md"
    )  # docs/reference/foo/bar.md (virtual)
    full_doc_path = Path("reference") / rel.with_suffix(".md")

    nav[rel.parts] = full_doc_path.as_posix()

    with mkdocs_gen_files.open(doc_path, "w") as f:
        f.write(f"# `{module}`\n\n")
        f.write(f"::: {module}\n")

# Write a SUMMARY.md for literate-nav
with mkdocs_gen_files.open("SUMMARY.md", "w") as f:
    f.writelines(nav.build_literate_nav())
