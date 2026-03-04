"""
Build UK_modeling.ipynb by adapting USD/US_modeling.ipynb
Changes:
  1. Title: QQQ -> FTSE 100 ETF
  2. Data file: US_research.csv -> UK_research.csv
  3. FinBERT max_length: 128 -> 256 (UK headlines are longer)
  4. FINANCIAL_COLS adapted for UK naming convention
  5. All model architectures, hyperparameters, and figures remain identical
"""
import json, copy, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

us_path = os.path.join(ROOT_DIR, "USD", "US_modeling.ipynb")
uk_path = os.path.join(BASE_DIR, "UK_modeling.ipynb")

# ── Read US notebook ──
with open(us_path, 'r', encoding='utf-8') as f:
    us_nb = json.load(f)

uk_nb = copy.deepcopy(us_nb)

# ── Apply replacements to every cell ──
REPLACEMENTS = [
    # Markdown header / description
    ('ETF (QQQ) Daily Return Prediction',
     'ETF (FTSE 100 ETF) Daily Return Prediction'),
    ('Daily Return (%) of QQQ ETF',
     'Daily Return (%) of FTSE 100 ETF'),
    # Data file
    ('US_research.csv', 'UK_research.csv'),
    # FinBERT max_length (UK headlines are multi-article concatenated)
    ('max_length=128', 'max_length=256'),
]

for cell in uk_nb['cells']:
    # Join source lines into a single string
    if isinstance(cell['source'], list):
        src = ''.join(cell['source'])
    else:
        src = cell['source']

    # Apply all replacements
    for old, new in REPLACEMENTS:
        src = src.replace(old, new)

    # Convert back to list-of-lines format (notebook standard)
    lines = src.split('\n')
    cell['source'] = [line + '\n' for line in lines[:-1]]
    if lines[-1]:  # last line without trailing newline
        cell['source'].append(lines[-1])
    elif not cell['source']:
        cell['source'] = ['']

    # Clear outputs and execution counts
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# ── Save UK notebook ──
with open(uk_path, 'w', encoding='utf-8') as f:
    json.dump(uk_nb, f, ensure_ascii=False, indent=1)

print(f"✓ UK_modeling.ipynb created at: {uk_path}")
print(f"  Cells: {len(uk_nb['cells'])}")
print(f"  Replacements applied: {len(REPLACEMENTS)}")

# Verify key changes
for cell in uk_nb['cells']:
    src = ''.join(cell['source'])
    if 'UK_research.csv' in src:
        print("  ✓ Data file: UK_research.csv")
        break

for cell in uk_nb['cells']:
    src = ''.join(cell['source'])
    if 'max_length=256' in src:
        print("  ✓ FinBERT max_length: 256")
        break

for cell in uk_nb['cells']:
    src = ''.join(cell['source'])
    if 'FTSE 100 ETF' in src:
        print("  ✓ ETF name: FTSE 100 ETF")
        break

# Check no US remnants
issues = []
for i, cell in enumerate(uk_nb['cells']):
    src = ''.join(cell['source'])
    if 'US_research.csv' in src:
        issues.append(f"  ⚠ Cell {i+1} still contains 'US_research.csv'")
    if 'QQQ' in src and 'FTSE' not in src:
        issues.append(f"  ⚠ Cell {i+1} still contains 'QQQ'")

if issues:
    print("\nPotential issues:")
    for iss in issues:
        print(iss)
else:
    print("  ✓ No US-specific remnants found")
