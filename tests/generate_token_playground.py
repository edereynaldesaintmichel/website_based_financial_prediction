"""Generate an HTML playground visualizing 2D RoPE tokenization."""
import sys, json, os, re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from table_financial_bert import (
    TableFinancialBertTokenizer, parse_table_grid, _TABLE_RE,
    TABLE_START_ID, TABLE_END_ID, TAB_ID,
)


def main():
    input_path = "/Users/eloireynal/Documents/My projects/website_based_financial_prediction/training_data/processed/companies_house_markdown_tagged/Prod224_0050_00222839_20170928.md"
    with open(input_path) as f:
        raw_text = f.read()

    tokenizer = TableFinancialBertTokenizer()
    result = tokenizer._tokenize_single(raw_text, add_special_tokens=True)

    # --- Build token metadata for the playground ---
    # Walk the same region structure as _tokenize_single to recover
    # token text, newline positions, and table row/col info.

    regions = []
    prev = 0
    for m in _TABLE_RE.finditer(raw_text):
        if m.start() > prev:
            regions.append(('text', raw_text[prev:m.start()], prev))
        regions.append(('table', m.group(0), m.start()))
        prev = m.end()
    if prev < len(raw_text):
        regions.append(('text', raw_text[prev:], prev))

    base_tok = tokenizer.base_tokenizer
    tokens = []
    tok_cursor = 0
    has_cls = (result["input_ids"][0] == base_tok.cls_token_id)

    # CLS token
    if has_cls:
        tokens.append({
            "index": 0, "token_id": result["input_ids"][0],
            "text": "[CLS]", "is_number": False,
            "number_sign": 0, "number_magnitude": 0.0,
            "col_pos": round(result["position_ids_col"][0], 4),
            "row_pos": round(result["position_ids_row"][0], 4),
            "newlines_before": 0, "region": "special",
        })
        tok_cursor = 1

    prev_region_end = 0

    for rtype, content, abs_start in regions:
        gap_text = raw_text[prev_region_end:abs_start]
        gap_newlines = gap_text.count("\n")

        if rtype == 'text':
            # Tokenize with offset_mapping for newline counting
            processed, num_positions = _replace_numbers(content, tokenizer)
            encoded = base_tok(
                processed, add_special_tokens=False,
                return_offsets_mapping=True, return_tensors=None,
            )
            offsets = encoded["offset_mapping"]
            orig_ids = encoded["input_ids"]
            n = len(orig_ids)

            for j in range(n):
                tid = result["input_ids"][tok_cursor]
                is_num = bool(result["is_number_mask"][tok_cursor])
                num_val = result["number_values"][tok_cursor]

                if is_num:
                    token_str = f"[NUM mag={num_val:.2f}]"
                else:
                    token_str = base_tok.decode([orig_ids[j]])

                s, e = offsets[j]
                nl = 0
                if j == 0:
                    nl += gap_newlines
                    gap_newlines = 0
                nl += processed[s:e].count("\n") if e > s else 0
                if j > 0:
                    prev_e = offsets[j - 1][1]
                    nl += processed[prev_e:s].count("\n")

                tokens.append({
                    "index": tok_cursor, "token_id": tid,
                    "text": token_str, "is_number": is_num,
                    "number_magnitude": round(num_val, 4),
                    "col_pos": round(result["position_ids_col"][tok_cursor], 4),
                    "row_pos": round(result["position_ids_row"][tok_cursor], 4),
                    "newlines_before": nl, "region": "text",
                })
                tok_cursor += 1

            prev_region_end = abs_start + len(content)

        else:  # table
            grid = parse_table_grid(content)
            first_cell = True

            # TABLE_START marker
            tokens.append({
                "index": tok_cursor, "token_id": TABLE_START_ID,
                "text": "[TABLE]", "is_number": False,
                "col_pos": round(result["position_ids_col"][tok_cursor], 4),
                "row_pos": round(result["position_ids_row"][tok_cursor], 4),
                "newlines_before": gap_newlines, "region": "special",
            })
            gap_newlines = 0
            tok_cursor += 1

            for ri, row_cells in enumerate(grid):
                for ci, cell_html in row_cells:
                    seg = tokenizer._tokenize_single(
                        cell_html, add_special_tokens=False
                    )
                    # Get original token ids for decoding
                    processed, _ = _replace_numbers(cell_html, tokenizer)
                    orig_ids = base_tok(
                        processed, add_special_tokens=False, return_tensors=None
                    )["input_ids"]
                    n = len(seg["input_ids"])

                    for j in range(n):
                        tid = result["input_ids"][tok_cursor]
                        is_num = bool(result["is_number_mask"][tok_cursor])
                        num_val = result["number_values"][tok_cursor]

                        if is_num:
                            token_str = f"[NUM mag={num_val:.2f}]"
                        else:
                            token_str = base_tok.decode([orig_ids[j]])

                        nl = 0
                        if first_cell and j == 0:
                            first_cell = False

                        tokens.append({
                            "index": tok_cursor, "token_id": tid,
                            "text": token_str, "is_number": is_num,
                            "number_magnitude": round(num_val, 4),
                            "col_pos": round(result["position_ids_col"][tok_cursor], 4),
                            "row_pos": round(result["position_ids_row"][tok_cursor], 4),
                            "newlines_before": nl, "region": "table",
                            "table_row": ri, "table_col": ci,
                        })
                        tok_cursor += 1

                    # TAB delimiter after cell
                    tokens.append({
                        "index": tok_cursor, "token_id": TAB_ID,
                        "text": "\\t", "is_number": False,
                        "number_sign": 0, "number_magnitude": 0.0,
                        "col_pos": round(result["position_ids_col"][tok_cursor], 4),
                        "row_pos": round(result["position_ids_row"][tok_cursor], 4),
                        "newlines_before": 0, "region": "table",
                        "table_row": ri, "table_col": ci,
                    })
                    tok_cursor += 1

            # TABLE_END marker
            tokens.append({
                "index": tok_cursor, "token_id": TABLE_END_ID,
                "text": "[/TABLE]", "is_number": False,
                "number_sign": 0, "number_magnitude": 0.0,
                "col_pos": round(result["position_ids_col"][tok_cursor], 4),
                "row_pos": round(result["position_ids_row"][tok_cursor], 4),
                "newlines_before": 0, "region": "special",
            })
            tok_cursor += 1

            prev_region_end = abs_start + len(content)

    # SEP token
    if tok_cursor < len(result["input_ids"]):
        tokens.append({
            "index": tok_cursor, "token_id": result["input_ids"][tok_cursor],
            "text": "[SEP]", "is_number": False,
            "number_sign": 0, "number_magnitude": 0.0,
            "col_pos": round(result["position_ids_col"][tok_cursor], 4),
            "row_pos": round(result["position_ids_row"][tok_cursor], 4),
            "newlines_before": 0, "region": "special",
        })

    output_path = os.path.join(os.path.dirname(__file__), "token_playground.html")
    with open(output_path, "w") as f:
        f.write(generate_html(tokens, raw_text))
    print(f"Written to {output_path}")


def _replace_numbers(text, tokenizer):
    """Replace <number>...</number> tags with § placeholder. Returns (processed, char_positions)."""
    processed = text
    positions = []
    offset = 0
    for match in tokenizer.number_pattern.finditer(text):
        start = match.start() + offset
        end = match.end() + offset
        processed = processed[:start] + "\u00a7" + processed[end:]
        positions.append(start)
        offset += 1 - (match.end() - match.start())
    return processed, positions


def generate_html(tokens, original_text):
    tokens_json = json.dumps(tokens)
    escaped_text = (
        original_text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>2D RoPE Token Playground</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
h1 {{ font-size: 1.4rem; margin-bottom: 6px; color: #e0e0ff; }}
.subtitle {{ font-size: 0.85rem; color: #888; margin-bottom: 20px; }}

.container {{ display: flex; gap: 20px; height: calc(100vh - 100px); }}
.token-panel {{ flex: 1; overflow-y: auto; }}
.info-panel {{ width: 340px; min-width: 340px; }}

.section {{ background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
.section h2 {{ font-size: 1rem; color: #a0a0ff; margin-bottom: 10px; }}

.token-flow {{ line-height: 2.2; }}
.token {{
    display: inline-block;
    padding: 2px 5px;
    margin: 1px;
    border-radius: 4px;
    cursor: pointer;
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 0.82rem;
    transition: transform 0.1s, box-shadow 0.1s;
    border: 2px solid transparent;
    white-space: pre-wrap;
}}
.token:hover {{
    transform: scale(1.12);
    box-shadow: 0 0 10px rgba(255,255,255,0.3);
    z-index: 10;
    position: relative;
}}
.token.selected {{
    border-color: #fff;
    box-shadow: 0 0 14px rgba(255,255,255,0.5);
}}
.token.is-number {{
    font-weight: bold;
    text-decoration: underline;
    text-decoration-style: dotted;
}}

.info-card {{ background: #0f3460; border-radius: 8px; padding: 16px; }}
.info-card .label {{ color: #7878c0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; }}
.info-card .value {{ font-size: 1.05rem; margin-bottom: 12px; font-family: 'Fira Code', monospace; }}
.info-card .value.big {{ font-size: 1.3rem; color: #fff; }}
.info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; }}

.original-text {{ font-family: 'Fira Code', monospace; font-size: 0.8rem; line-height: 1.6; color: #ccc; max-height: 200px; overflow-y: auto; padding: 10px; background: #0d1b33; border-radius: 6px; }}

.legend {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }}
.legend-item {{ font-size: 0.75rem; padding: 2px 8px; border-radius: 3px; }}

.pos-chart {{ width: 100%; aspect-ratio: 1 / 1; background: #0d1b33; border-radius: 6px; position: relative; overflow: hidden; }}
.pos-dot {{
    position: absolute; width: 6px; height: 6px; border-radius: 50%;
    transform: translate(-50%, -50%); opacity: 0.6; transition: opacity 0.15s;
}}
.pos-dot.highlighted {{ opacity: 1; width: 10px; height: 10px; box-shadow: 0 0 8px rgba(255,255,255,0.5); }}
.axis-label {{ position: absolute; font-size: 0.65rem; color: #666; }}
</style>
</head>
<body>

<h1>2D RoPE Token Playground</h1>
<p class="subtitle">Click any token to inspect. Colors cycle per token index. Numbers are underlined.</p>

<div class="container">
  <div class="token-panel">
    <div class="section">
      <h2>Tokens</h2>
      <div class="legend" id="legend"></div>
      <div class="token-flow" id="token-flow"></div>
    </div>
    <div class="section">
      <h2>2D Position Map (col vs row)</h2>
      <div class="pos-chart" id="pos-chart">
        <span class="axis-label" style="bottom:2px;left:50%;">col position &rarr;</span>
        <span class="axis-label" style="top:50%;left:2px;transform:rotate(-90deg) translateX(-50%);transform-origin:left top;">row position &rarr;</span>
      </div>
    </div>
    <div class="section">
      <h2>Original Text (escaped)</h2>
      <div class="original-text">{escaped_text}</div>
    </div>
  </div>

  <div class="info-panel">
    <div class="section">
      <h2>Token Info</h2>
      <div class="info-card" id="info-card">
        <p style="color:#666;">Click a token to see details.</p>
      </div>
    </div>
  </div>
</div>

<script>
const TOKENS = {tokens_json};

const COLORS = [
  '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
  '#1abc9c', '#e67e22', '#e84393', '#00cec9', '#fd79a8'
];

function escapeHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// Render tokens
const flow = document.getElementById('token-flow');
let prevRow = -1, prevCol = -1, prevRegion = '';
TOKENS.forEach((tok, i) => {{
  // Insert line breaks to preserve original text layout
  for (let n = 0; n < (tok.newlines_before || 0); n++) {{
    flow.appendChild(document.createElement('br'));
  }}

  // Table cell/row separators
  if (tok.region === 'table') {{
    if (prevRegion !== 'table') {{
      const marker = document.createElement('span');
      marker.style.cssText = 'color:#555;font-size:0.7rem;margin:0 4px;';
      marker.textContent = '\u2503';
      flow.appendChild(marker);
    }} else if (tok.table_row !== prevRow) {{
      flow.appendChild(document.createElement('br'));
      const marker = document.createElement('span');
      marker.style.cssText = 'color:#555;font-size:0.7rem;margin:0 2px;';
      marker.textContent = '\u2503';
      flow.appendChild(marker);
    }} else if (tok.table_col !== prevCol) {{
      const sep = document.createElement('span');
      sep.style.cssText = 'color:#444;font-size:0.7rem;margin:0 2px;';
      sep.textContent = '\u2502';
      flow.appendChild(sep);
    }}
    prevRow = tok.table_row;
    prevCol = tok.table_col;
  }} else if (prevRegion === 'table') {{
    const marker = document.createElement('span');
    marker.style.cssText = 'color:#555;font-size:0.7rem;margin:0 4px;';
    marker.textContent = '\u2503';
    flow.appendChild(marker);
  }}
  prevRegion = tok.region;

  const span = document.createElement('span');
  span.className = 'token' + (tok.is_number ? ' is-number' : '');
  const bg = COLORS[i % COLORS.length];
  span.style.background = bg + '30';
  span.style.color = bg;
  span.dataset.index = i;

  let display = tok.text;
  if (display === '' || display === ' ') display = tok.text === ' ' ? '\u00b7' : '\u2205';
  span.textContent = display;
  span.title = `#${{i}} id=${{tok.token_id}} col=${{tok.col_pos}} row=${{tok.row_pos}}`;
  span.addEventListener('click', () => selectToken(i));
  flow.appendChild(span);
}});

// Render position chart
const chart = document.getElementById('pos-chart');
const colPositions = TOKENS.map(t => t.col_pos);
const rowPositions = TOKENS.map(t => t.row_pos);
const minCol = Math.min(...colPositions), maxCol = Math.max(...colPositions);
const minRow = Math.min(...rowPositions), maxRow = Math.max(...rowPositions);
const rangeCol = maxCol - minCol || 1, rangeRow = maxRow - minRow || 1;

TOKENS.forEach((tok, i) => {{
  const dot = document.createElement('div');
  dot.className = 'pos-dot';
  dot.dataset.index = i;
  const x = ((tok.col_pos - minCol) / rangeCol) * 90 + 5;
  const y = ((tok.row_pos - minRow) / rangeRow) * 85 + 7;
  dot.style.left = x + '%';
  dot.style.top = y + '%';
  dot.style.background = COLORS[i % COLORS.length];
  dot.addEventListener('click', () => selectToken(i));
  chart.appendChild(dot);
}});

// Legend
const legend = document.getElementById('legend');
COLORS.forEach((c, i) => {{
  const item = document.createElement('span');
  item.className = 'legend-item';
  item.style.background = c + '30';
  item.style.color = c;
  item.textContent = `${{i}}`;
  legend.appendChild(item);
}});

let selectedIndex = -1;

function selectToken(i) {{
  const tok = TOKENS[i];
  document.querySelectorAll('.token.selected').forEach(el => el.classList.remove('selected'));
  document.querySelectorAll('.pos-dot.highlighted').forEach(el => el.classList.remove('highlighted'));

  const tokenEls = document.querySelectorAll('.token');
  const dotEls = document.querySelectorAll('.pos-dot');
  if (tokenEls[i]) tokenEls[i].classList.add('selected');
  if (dotEls[i]) dotEls[i].classList.add('highlighted');

  selectedIndex = i;

  const card = document.getElementById('info-card');
  card.innerHTML = `
    <div class="label">Token Text</div>
    <div class="value big">${{escapeHtml(tok.text)}}</div>
    <div class="info-grid">
      <div><div class="label">Index</div><div class="value">${{tok.index}}</div></div>
      <div><div class="label">Token ID</div><div class="value">${{tok.token_id}}</div></div>
      <div><div class="label">Col Position</div><div class="value">${{tok.col_pos}}</div></div>
      <div><div class="label">Row Position</div><div class="value">${{tok.row_pos}}</div></div>
      <div><div class="label">Region</div><div class="value">${{tok.region}}${{tok.region === 'table' ? ' [r' + tok.table_row + ', c' + tok.table_col + ']' : ''}}</div></div>
      <div><div class="label">Is Number</div><div class="value">${{tok.is_number ? 'Yes' : 'No'}}</div></div>
      ${{tok.is_number ? `
        <div><div class="label">Sign</div><div class="value">${{tok.number_sign === 0 ? 'Positive' : 'Negative'}}</div></div>
        <div><div class="label">Magnitude (log\u2081\u2080)</div><div class="value">${{tok.number_magnitude}}</div></div>
      ` : ''}}
      <div><div class="label">Color Index</div><div class="value">${{i % COLORS.length}}</div></div>
    </div>
  `;
}}
</script>
</body>
</html>""";


if __name__ == "__main__":
    main()
