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

    // Pass 1: remove empty columns
    let colCount = Math.max(0, ...grid.map(r => r.length));
    for (let col = colCount - 1; col >= 0; col--) {
        if (colMatches(col, /^$/)) removeCol(col);
    }

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
});