# pytest --cov=fdm_edl --cov-report=html --cov-report=term-missing tests/
mkdir -p docs/codecov
cp -r htmlcov/* docs/codecov/
rm -rf htmlcov
mkdocs build
# mkdocs serve
