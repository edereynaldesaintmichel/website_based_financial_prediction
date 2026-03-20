const strip = s => s.replace(/[\u200B\u200C\u200D\uFEFF\u00A0]/g, ' ').trim();
const PREFIXES = /^[\$\(]+$/;
const SUFFIXES = /^[\)\%]+$/;

// Initial cleanup — textContent avoids reflows entirely
const cells = [...document.querySelectorAll('td,th')];
cells.forEach(x => { x.textContent = strip(x.textContent); x.removeAttribute('style'); });
[...document.getElementsByTagName('tr')].forEach(tr => {
    if (strip(tr.textContent) === '') tr.remove();
});

document.querySelectorAll('table').forEach(table => {
    const rows = [...table.querySelectorAll('tr')];
    if (rows.length === 0) return;

    // Build grid once, cache cell text (avoids reflows)
    const grid = [];
    const text = new Map();
    rows.forEach((tr, ri) => {
        if (!grid[ri]) grid[ri] = [];
        [...tr.cells].forEach(cell => {
            let c = 0;
            while (grid[ri][c]) c++;
            const cs = cell.colSpan || 1;
            const rs = cell.rowSpan || 1;
            if (!text.has(cell)) text.set(cell, cell.textContent.trim());
            for (let dr = 0; dr < rs; dr++) {
                if (!grid[ri + dr]) grid[ri + dr] = [];
                for (let dc = 0; dc < cs; dc++) {
                    grid[ri + dr][c + dc] = { cell, isOrigin: dr === 0 && dc === 0 };
                }
            }
        });
    });

    const T = cell => text.get(cell) || '';

    function dumpGrid(label) {
        console.log(`\n=== ${label} (${grid.length} rows x ${Math.max(0,...grid.map(r=>r.length))} cols) ===`);
        grid.forEach((r, ri) => {
            const cells = [];
            for (let c = 0; c < r.length; c++) {
                if (!r[c]) { cells.push('_'); continue; }
                const t = T(r[c].cell);
                const cs = r[c].cell.colSpan;
                const o = r[c].isOrigin ? 'O' : '.';
                cells.push(`${o}${cs > 1 ? 'c'+cs : ''}[${t}]`);
            }
            console.log(`  r${ri}: ${cells.join(' | ')}`);
        });
    }

    //dumpGrid('After grid build');

    // Fix overflow rows: when rowspan cells from a previous row fill all positions,
    // the current row's own cells get pushed beyond the expected grid width.
    // Detect this and re-map: decrement rowspans, put own cells at positions 0..N.
    {
        // Expected width = most common row length (mode)
        const lengths = grid.map(r => r.length);
        const freq = {};
        lengths.forEach(l => freq[l] = (freq[l] || 0) + 1);
        const expectedWidth = +Object.entries(freq).sort((a, b) => b[1] - a[1])[0][0];

        grid.forEach((r, ri) => {
            if (r.length <= expectedWidth) return;
            // Collect origin cells in this row (these are the row's "own" cells)
            const ownCells = [];
            for (let c = 0; c < r.length; c++) {
                if (r[c] && r[c].isOrigin) ownCells.push(r[c]);
            }
            // Decrement rowspan for cells from other rows that extend into this row
            const seen = new Set();
            for (let c = 0; c < r.length; c++) {
                if (r[c] && !r[c].isOrigin && !seen.has(r[c].cell)) {
                    seen.add(r[c].cell);
                    if (r[c].cell.rowSpan > 1) r[c].cell.rowSpan--;
                }
            }
            // Rebuild row: place own cells at positions 0..N
            grid[ri] = [];
            ownCells.forEach((entry, c) => {
                grid[ri][c] = entry;
                // Also fill colspan/rowspan slots
                const cs = entry.cell.colSpan || 1;
                const rs = entry.cell.rowSpan || 1;
                for (let dc = 1; dc < cs; dc++) {
                    grid[ri][c + dc] = { cell: entry.cell, isOrigin: false };
                }
                for (let dr = 1; dr < rs; dr++) {
                    if (!grid[ri + dr]) grid[ri + dr] = [];
                    for (let dc = 0; dc < cs; dc++) {
                        grid[ri + dr][c + dc] = { cell: entry.cell, isOrigin: false };
                    }
                }
            });
        });
    }

    //dumpGrid('After overflow fix');

    function removeCol(col) {
        grid.forEach(r => {
            if (!r[col]) return;
            const { cell, isOrigin } = r[col];
            if (cell.colSpan > 1) cell.colSpan--;
            else if (isOrigin) cell.remove();
        });
        grid.forEach(r => r.splice(col, 1));
    }

    function colMatches(col, pattern) {
        return grid.every(r => {
            if (!r[col]) return true;
            const t = T(r[col].cell);
            return t === '' || pattern.test(t) || r[col].cell.colSpan > 1;
        });
    }

    // Pass 0: factorize colspans — if two adjacent visual columns are always
    // occupied by the same cell in every row, they're redundant. Collapse them.
    for (let col = Math.max(0, ...grid.map(r => r.length)) - 1; col >= 1; col--) {
        const canMerge = grid.every(r =>
            (!r[col] && !r[col - 1]) ||
            (r[col] && r[col - 1] && r[col].cell === r[col - 1].cell)
        );
        if (canMerge) {
            // Decrement colspan for cells spanning this redundant column
            const seen = new Set();
            grid.forEach(r => {
                if (!r[col]) return;
                if (seen.has(r[col].cell)) return;
                seen.add(r[col].cell);
                if (r[col].cell.colSpan > 1) r[col].cell.colSpan--;
            });
            grid.forEach(r => r.splice(col, 1));
        }
    }

    //dumpGrid('After factorize');

    // Pass 0b: split colspan cells where other rows have distinct cells.
    // For each colspan>1 cell, check if any row has distinct (colspan=1) cells
    // at those visual positions. If so, split the cell and place its text in
    // the sub-column where other rows have the most non-empty content.
    {
        const colCount = Math.max(0, ...grid.map(r => r.length));
        // Pre-compute: for each column, count distinct cells with "substantial"
        // content (not just prefix/suffix tokens like $ ( ) %)
        const nonEmptyCount = [];
        for (let col = 0; col < colCount; col++) {
            let count = 0;
            grid.forEach(r => {
                if (r[col] && r[col].isOrigin && r[col].cell.colSpan === 1) {
                    const t = T(r[col].cell);
                    if (t && !PREFIXES.test(t) && !SUFFIXES.test(t)) count++;
                }
            });
            nonEmptyCount[col] = count;
        }

        for (let col = 0; col < colCount; col++) {
            grid.forEach((r, ri) => {
                if (!r[col] || !r[col].isOrigin) return;
                const { cell } = r[col];
                if (cell.colSpan <= 1) return;
                const cs = cell.colSpan;
                const rs = cell.rowSpan || 1;

                // Check if any sub-column has distinct cells in other rows
                let needsSplit = false;
                for (let dc = 0; dc < cs; dc++) {
                    if (nonEmptyCount[col + dc] > 0) { needsSplit = true; break; }
                }
                if (!needsSplit) return;

                // Find best sub-column for the text: the one where other rows
                // have the most non-prefix/suffix content. On ties, prefer
                // rightmost (left columns tend to be prefixes/spacers).
                let bestDc = cs - 1, bestCount = -1;
                for (let dc = cs - 1; dc >= 0; dc--) {
                    if (nonEmptyCount[col + dc] > bestCount) {
                        bestCount = nonEmptyCount[col + dc];
                        bestDc = dc;
                    }
                }

                const cellText = T(cell);
                cell.colSpan = 1;
                // Create new cells for sub-columns 1..cs-1 (inserted after origin)
                let insertAfter = cell;
                for (let dc = 1; dc < cs; dc++) {
                    const newCell = document.createElement(cell.tagName);
                    if (rs > 1) newCell.rowSpan = rs;
                    if (dc === bestDc) {
                        newCell.textContent = cellText;
                        text.set(newCell, cellText);
                    } else {
                        newCell.textContent = '';
                        text.set(newCell, '');
                    }
                    insertAfter.after(newCell);
                    insertAfter = newCell;
                    for (let dr = 0; dr < rs; dr++) {
                        if (grid[ri + dr]) {
                            grid[ri + dr][col + dc] = { cell: newCell, isOrigin: dr === 0 };
                        }
                    }
                }
                // If text moved to a new cell, clear the origin
                if (bestDc !== 0) {
                    cell.textContent = '';
                    text.set(cell, '');
                }
            });
        }
    }

    //dumpGrid('After split');

    // Pass 1: remove empty columns
    let colCount = Math.max(0, ...grid.map(r => r.length));
    for (let col = colCount - 1; col >= 0; col--) {
        if (colMatches(col, /^$/)) removeCol(col);
    }

    //dumpGrid('After empty removal');

    // Pass 2: merge prefix columns into the right neighbor
    colCount = Math.max(0, ...grid.map(r => r.length));
    for (let col = colCount - 2; col >= 0; col--) {
        if (!colMatches(col, PREFIXES)) continue;
        grid.forEach(r => {
            if (!r[col] || !r[col + 1]) return;
            const p = T(r[col].cell);
            if (p && r[col].cell.colSpan <= 1) {
                const merged = p + T(r[col + 1].cell);
                text.set(r[col + 1].cell, merged);
                r[col + 1].cell.textContent = merged;
            }
        });
        removeCol(col);
        colCount--;
    }

    //dumpGrid('After prefix merge');

    // Pass 3: merge suffix columns into the left neighbor
    colCount = Math.max(0, ...grid.map(r => r.length));
    for (let col = colCount - 1; col >= 1; col--) {
        if (!colMatches(col, SUFFIXES)) continue;
        grid.forEach(r => {
            if (!r[col] || !r[col - 1]) return;
            const s = T(r[col].cell);
            if (s && r[col].cell.colSpan <= 1) {
                const merged = T(r[col - 1].cell) + s;
                text.set(r[col - 1].cell, merged);
                r[col - 1].cell.textContent = merged;
            }
        });
        removeCol(col);
        colCount--;
    }
    //dumpGrid('After suffix merge');
});