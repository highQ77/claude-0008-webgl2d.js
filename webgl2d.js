/*!
 * webgl2d.js — 以 WebGL 實作的 CanvasRenderingContext2D 相容繪圖 API
 *
 * 用法：
 *   const ctx = WebGL2D.getContext(canvas);   // 取代 canvas.getContext('2d')
 *   ctx.fillStyle = 'tomato';
 *   ctx.fillRect(10, 10, 100, 50);
 *
 * 實作重點：
 *   - 填色：Stencil-then-Cover（模板緩衝計算 nonzero / evenodd 環繞數，再覆蓋上色）
 *   - 描邊：CPU 產生線段 / 接合 / 端點三角形，經模板去重疊後上色（半透明不會重疊加深）
 *   - 裁切：模板緩衝最高位元 (0x80) 作為裁切遮罩，支援巢狀 clip 與 save/restore
 *   - 漸層：Linear / Radial（雙圓錐形，完整符合規格）/ Conic，於 fragment shader 計算
 *   - 文字：使用離屏 2D canvas 只做字形點陣化成遮罩，再以 WebGL 套用任意 fillStyle
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WebGL2D = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const TAU = Math.PI * 2;
  const TOL = 0.2; // 曲線攤平容許誤差（裝置像素）

  /* =====================================================================
   * 例外
   * ===================================================================== */
  function domError(msg, name) {
    if (typeof DOMException === 'function') return new DOMException(msg, name);
    const e = new Error(msg); e.name = name; return e;
  }
  const fin = (...a) => a.every(Number.isFinite);

  /* =====================================================================
   * 2D 仿射矩陣 [a, b, c, d, e, f]：x' = a·x + c·y + e, y' = b·x + d·y + f
   * ===================================================================== */
  function mul(m, n) {
    return [
      m[0] * n[0] + m[2] * n[1],
      m[1] * n[0] + m[3] * n[1],
      m[0] * n[2] + m[2] * n[3],
      m[1] * n[2] + m[3] * n[3],
      m[0] * n[4] + m[2] * n[5] + m[4],
      m[1] * n[4] + m[3] * n[5] + m[5],
    ];
  }
  function inv(m) {
    const det = m[0] * m[3] - m[1] * m[2];
    if (!det || !Number.isFinite(det)) return null;
    return [
      m[3] / det, -m[1] / det, -m[2] / det, m[0] / det,
      (m[2] * m[5] - m[3] * m[4]) / det,
      (m[1] * m[4] - m[0] * m[5]) / det,
    ];
  }
  const maxScale = (m) => Math.sqrt(Math.max(m[0] * m[0] + m[1] * m[1], m[2] * m[2] + m[3] * m[3]));
  function toMatrix(t) {
    if (!t) return [1, 0, 0, 1, 0, 0];
    if (Array.isArray(t)) return t.slice(0, 6);
    const g = (k, alt, def) => (t[k] !== undefined ? t[k] : t[alt] !== undefined ? t[alt] : def);
    const m = [g('a', 'm11', 1), g('b', 'm12', 0), g('c', 'm21', 0), g('d', 'm22', 1), g('e', 'm41', 0), g('f', 'm42', 0)];
    if (!fin(...m)) throw new TypeError('Matrix values must be finite');
    return m;
  }
  function makeDOMMatrix(m) {
    if (typeof DOMMatrix === 'function') return new DOMMatrix(m);
    return { a: m[0], b: m[1], c: m[2], d: m[3], e: m[4], f: m[5], is2D: true };
  }

  /* =====================================================================
   * CSS 顏色解析 → [r(0-255), g, b, a(0-1)]
   * ===================================================================== */
  const BASIC = {
    transparent: [0, 0, 0, 0], black: [0, 0, 0, 1], white: [255, 255, 255, 1], red: [255, 0, 0, 1],
    lime: [0, 255, 0, 1], green: [0, 128, 0, 1], blue: [0, 0, 255, 1], yellow: [255, 255, 0, 1],
    cyan: [0, 255, 255, 1], aqua: [0, 255, 255, 1], magenta: [255, 0, 255, 1], fuchsia: [255, 0, 255, 1],
    gray: [128, 128, 128, 1], grey: [128, 128, 128, 1], orange: [255, 165, 0, 1], purple: [128, 0, 128, 1],
    currentcolor: [0, 0, 0, 1],
  };
  const colorCache = new Map();
  let probeEl = null;

  function parseHex(h) {
    if (!/^[0-9a-f]+$/.test(h)) return null;
    const n = h.length;
    if (n === 3 || n === 4) {
      const v = [...h].map((c) => parseInt(c + c, 16));
      return [v[0], v[1], v[2], n === 4 ? v[3] / 255 : 1];
    }
    if (n === 6 || n === 8) {
      const v = [0, 2, 4, 6].map((i) => parseInt(h.substr(i, 2), 16));
      return [v[0], v[1], v[2], n === 8 ? v[3] / 255 : 1];
    }
    return null;
  }
  function fnArgs(s) {
    const open = s.indexOf('('), close = s.lastIndexOf(')');
    if (open < 0 || close < open) return null;
    return s.slice(open + 1, close).split(/[\s,\/]+/).filter(Boolean);
  }
  function parseAlpha(t) {
    if (t === undefined) return 1;
    if (t === 'none') return 0;
    const v = t.endsWith('%') ? parseFloat(t) / 100 : parseFloat(t);
    return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : NaN;
  }
  function parseRgb(s) {
    const a = fnArgs(s);
    if (!a || a.length < 3) return null;
    const ch = a.slice(0, 3).map((t) => {
      if (t === 'none') return 0;
      const v = t.endsWith('%') ? parseFloat(t) * 2.55 : parseFloat(t);
      return Math.round(Math.min(255, Math.max(0, v)));
    });
    const al = parseAlpha(a[3]);
    if (ch.some(Number.isNaN) || Number.isNaN(al)) return null;
    return [ch[0], ch[1], ch[2], al];
  }
  function parseHsl(s) {
    const a = fnArgs(s);
    if (!a || a.length < 3) return null;
    let h = parseFloat(a[0]);
    if (a[0].endsWith('rad')) h = (h * 180) / Math.PI;
    else if (a[0].endsWith('turn')) h *= 360;
    else if (a[0].endsWith('grad')) h *= 0.9;
    const sat = Math.min(1, Math.max(0, parseFloat(a[1]) / 100));
    const lig = Math.min(1, Math.max(0, parseFloat(a[2]) / 100));
    const al = parseAlpha(a[3]);
    if (!fin(h, sat, lig) || Number.isNaN(al)) return null;
    h = ((h % 360) + 360) % 360;
    const f = (n) => {
      const k = (n + h / 30) % 12;
      const c = lig - sat * Math.min(lig, 1 - lig) * Math.max(-1, Math.min(k - 3, 9 - k, 1));
      return Math.round(c * 255);
    };
    return [f(0), f(8), f(4), al];
  }
  function parseViaDOM(str) {
    if (typeof document === 'undefined' || !document.body) return null;
    if (!probeEl) {
      probeEl = document.createElement('span');
      probeEl.style.display = 'none';
      document.body.appendChild(probeEl);
    }
    probeEl.style.color = '';
    probeEl.style.color = str;
    if (!probeEl.style.color) return null;
    const cs = getComputedStyle(probeEl).color;
    return /^rgba?\(/.test(cs) ? parseRgb(cs) : null;
  }
  function parseColor(input) {
    if (typeof input !== 'string') return null;
    if (colorCache.has(input)) return colorCache.get(input);
    const s = input.trim().toLowerCase();
    let r = null;
    if (s[0] === '#') r = parseHex(s.slice(1));
    else if (/^rgba?\(/.test(s)) r = parseRgb(s);
    else if (/^hsla?\(/.test(s)) r = parseHsl(s);
    else if (BASIC[s]) r = BASIC[s].slice();
    else r = parseViaDOM(input);
    if (colorCache.size > 512) colorCache.clear();
    colorCache.set(input, r);
    return r;
  }
  function serializeColor(c) {
    const hx = (v) => v.toString(16).padStart(2, '0');
    if (c[3] >= 1) return '#' + hx(c[0]) + hx(c[1]) + hx(c[2]);
    const a = c[3] === 0 ? '0' : String(parseFloat(c[3].toFixed(3)));
    return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;
  }

  /* =====================================================================
   * 攤平後的路徑（裝置座標）
   *   subs: [{ pts: [x0,y0,x1,y1,...], closed: bool }]
   * 所有方法的第一個參數 m 為建構當下的變換矩陣（符合 Canvas 規格：點在加入時就被變換）
   * ===================================================================== */
  class FlatPath {
    /** pr：pixelRatio，用來決定曲線攤平的精細度（容許誤差以實體像素計） */
    constructor(pr = 1) { this.subs = []; this.cur = null; this.pr = pr; }
    _begin(x, y) { this.cur = { pts: [x, y], closed: false }; this.subs.push(this.cur); }
    _last() { const p = this.cur.pts; return [p[p.length - 2], p[p.length - 1]]; }
    _tx(m, x, y) { return [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]; }

    moveTo(m, x, y) {
      if (!fin(x, y)) return;
      const p = this._tx(m, x, y); this._begin(p[0], p[1]);
    }
    lineTo(m, x, y) {
      if (!fin(x, y)) return;
      if (!this.cur) return this.moveTo(m, x, y);
      const p = this._tx(m, x, y); this.cur.pts.push(p[0], p[1]);
    }
    closePath() {
      if (!this.cur) return;
      this.cur.closed = true;
      this._begin(this.cur.pts[0], this.cur.pts[1]);
    }
    quadraticCurveTo(m, cpx, cpy, x, y) {
      if (!fin(cpx, cpy, x, y)) return;
      if (!this.cur) this.moveTo(m, cpx, cpy);
      const [x0, y0] = this._last(), [x1, y1] = this._tx(m, cpx, cpy), [x2, y2] = this._tx(m, x, y);
      const M = Math.hypot(x0 - 2 * x1 + x2, y0 - 2 * y1 + y2);
      const n = Math.min(1024, Math.max(1, Math.ceil(Math.sqrt((0.25 * M * this.pr) / TOL))));
      const out = this.cur.pts;
      for (let i = 1; i <= n; i++) {
        const t = i / n, u = 1 - t;
        out.push(u * u * x0 + 2 * u * t * x1 + t * t * x2, u * u * y0 + 2 * u * t * y1 + t * t * y2);
      }
    }
    bezierCurveTo(m, c1x, c1y, c2x, c2y, x, y) {
      if (!fin(c1x, c1y, c2x, c2y, x, y)) return;
      if (!this.cur) this.moveTo(m, c1x, c1y);
      const [x0, y0] = this._last(), [x1, y1] = this._tx(m, c1x, c1y);
      const [x2, y2] = this._tx(m, c2x, c2y), [x3, y3] = this._tx(m, x, y);
      const M = Math.max(Math.hypot(x0 - 2 * x1 + x2, y0 - 2 * y1 + y2), Math.hypot(x1 - 2 * x2 + x3, y1 - 2 * y2 + y3));
      const n = Math.min(1024, Math.max(1, Math.ceil(Math.sqrt((0.75 * M * this.pr) / TOL))));
      const out = this.cur.pts;
      for (let i = 1; i <= n; i++) {
        const t = i / n, u = 1 - t;
        const a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
        out.push(a * x0 + b * x1 + c * x2 + d * x3, a * y0 + b * y1 + c * y2 + d * y3);
      }
    }
    ellipse(m, x, y, rx, ry, rot, sa, ea, ccw) {
      if (!fin(x, y, rx, ry, rot, sa, ea)) return;
      if (rx < 0 || ry < 0) throw domError('The radius provided is negative.', 'IndexSizeError');
      let sweep;
      if (!ccw && ea - sa >= TAU) sweep = TAU;
      else if (ccw && sa - ea >= TAU) sweep = -TAU;
      else {
        sweep = (ea - sa) % TAU;
        if (!ccw && sweep < 0) sweep += TAU;
        else if (ccw && sweep > 0) sweep -= TAU;
      }
      const r = Math.max(rx, ry) * maxScale(m) * this.pr;
      const da = r > TOL ? 2 * Math.acos(Math.max(-1, 1 - TOL / r)) : Math.PI / 2;
      const n = Math.min(4096, Math.max(1, Math.ceil(Math.abs(sweep) / da)));
      const cr = Math.cos(rot), sr = Math.sin(rot);
      for (let i = 0; i <= n; i++) {
        const t = sa + (sweep * i) / n, ct = Math.cos(t), st = Math.sin(t);
        const ux = x + rx * ct * cr - ry * st * sr;
        const uy = y + rx * ct * sr + ry * st * cr;
        if (i === 0) { if (this.cur) this.lineTo(m, ux, uy); else this.moveTo(m, ux, uy); }
        else { const p = this._tx(m, ux, uy); this.cur.pts.push(p[0], p[1]); }
      }
    }
    arc(m, x, y, r, sa, ea, ccw) { this.ellipse(m, x, y, r, r, 0, sa, ea, ccw); }
    arcTo(m, x1, y1, x2, y2, r) {
      if (!fin(x1, y1, x2, y2, r)) return;
      if (r < 0) throw domError('The radius provided is negative.', 'IndexSizeError');
      if (!this.cur) this.moveTo(m, x1, y1);
      const im = inv(m);
      if (!im) return;
      const [lx, ly] = this._last();
      const x0 = im[0] * lx + im[2] * ly + im[4], y0 = im[1] * lx + im[3] * ly + im[5];
      const v1x = x0 - x1, v1y = y0 - y1, v2x = x2 - x1, v2y = y2 - y1;
      const l1 = Math.hypot(v1x, v1y), l2 = Math.hypot(v2x, v2y);
      const cross = v1x * v2y - v1y * v2x;
      if (l1 < 1e-12 || l2 < 1e-12 || r === 0 || Math.abs(cross) < 1e-12 * l1 * l2) {
        this.lineTo(m, x1, y1); return;
      }
      const u1x = v1x / l1, u1y = v1y / l1, u2x = v2x / l2, u2y = v2y / l2;
      const ang = Math.acos(Math.max(-1, Math.min(1, u1x * u2x + u1y * u2y)));
      const dist = r / Math.tan(ang / 2);
      const t1x = x1 + u1x * dist, t1y = y1 + u1y * dist, t2x = x1 + u2x * dist, t2y = y1 + u2y * dist;
      let bx = u1x + u2x, by = u1y + u2y; const bl = Math.hypot(bx, by); bx /= bl; by /= bl;
      const cd = r / Math.sin(ang / 2), cx = x1 + bx * cd, cy = y1 + by * cd;
      const a0 = Math.atan2(t1y - cy, t1x - cx), a1 = Math.atan2(t2y - cy, t2x - cx);
      let d = a1 - a0;
      while (d > Math.PI) d -= TAU;
      while (d < -Math.PI) d += TAU;
      this.ellipse(m, cx, cy, r, r, 0, a0, a0 + d, d < 0);
    }
    rect(m, x, y, w, h) {
      if (!fin(x, y, w, h)) return;
      this.moveTo(m, x, y); this.lineTo(m, x + w, y); this.lineTo(m, x + w, y + h); this.lineTo(m, x, y + h);
      this.closePath();
    }
    roundRect(m, x, y, w, h, radii = 0) {
      if (!fin(x, y, w, h)) return;
      let list = Array.isArray(radii) || (radii && typeof radii[Symbol.iterator] === 'function' && typeof radii !== 'string') ? [...radii] : [radii];
      if (list.length < 1 || list.length > 4) throw new RangeError('roundRect radii must contain 1 to 4 values');
      const R = list.map((v) => {
        const p = typeof v === 'number' ? { x: v, y: v } : { x: +(v && v.x) || 0, y: +(v && v.y) || 0 };
        if (!fin(p.x, p.y)) return null;
        if (p.x < 0 || p.y < 0) throw new RangeError('roundRect radius is negative');
        return p;
      });
      if (R.includes(null)) return;
      let tl, tr, br, bl;
      if (R.length === 1) tl = tr = br = bl = R[0];
      else if (R.length === 2) { tl = br = R[0]; tr = bl = R[1]; }
      else if (R.length === 3) { tl = R[0]; tr = bl = R[1]; br = R[2]; }
      else [tl, tr, br, bl] = R;
      tl = { ...tl }; tr = { ...tr }; br = { ...br }; bl = { ...bl };
      if (w < 0) { x += w; w = -w; [tl, tr] = [tr, tl]; [bl, br] = [br, bl]; }
      if (h < 0) { y += h; h = -h; [tl, bl] = [bl, tl]; [tr, br] = [br, tr]; }
      const sc = Math.min(1,
        w / (tl.x + tr.x || Infinity), w / (bl.x + br.x || Infinity),
        h / (tl.y + bl.y || Infinity), h / (tr.y + br.y || Infinity));
      for (const c of [tl, tr, br, bl]) { c.x *= sc; c.y *= sc; }
      const H = Math.PI / 2;
      this.moveTo(m, x + tl.x, y);
      this.lineTo(m, x + w - tr.x, y);
      if (tr.x && tr.y) this.ellipse(m, x + w - tr.x, y + tr.y, tr.x, tr.y, 0, -H, 0, false);
      this.lineTo(m, x + w, y + h - br.y);
      if (br.x && br.y) this.ellipse(m, x + w - br.x, y + h - br.y, br.x, br.y, 0, 0, H, false);
      this.lineTo(m, x + bl.x, y + h);
      if (bl.x && bl.y) this.ellipse(m, x + bl.x, y + h - bl.y, bl.x, bl.y, 0, H, Math.PI, false);
      this.lineTo(m, x, y + tl.y);
      if (tl.x && tl.y) this.ellipse(m, x + tl.x, y + tl.y, tl.x, tl.y, 0, Math.PI, Math.PI + H, false);
      this.closePath();
    }
    addPath(m, ops, mat) { replay(this, ops, mat ? mul(m, mat) : m); }
    snapshot() { return this.subs.map((s) => ({ pts: s.pts.slice(), closed: s.closed })); }
  }

  function replay(flat, ops, m) {
    for (const op of ops) {
      const name = op[0];
      if (name === 'closePath') flat.closePath();
      else if (name === 'addPath') flat.addPath(m, op[1], op[2]);
      else flat[name](m, ...op.slice(1));
    }
  }

  /* =====================================================================
   * SVG path data 解析 → Path2D 指令
   * ===================================================================== */
  function parseSVGPath(d) {
    const ops = [];
    const NUM = /[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/y;
    let i = 0;
    const n = d.length;
    const skip = () => { while (i < n && /[\s,]/.test(d[i])) i++; };
    const num = () => {
      skip(); NUM.lastIndex = i;
      const r = NUM.exec(d);
      if (!r) throw 0;
      i = NUM.lastIndex; return parseFloat(r[0]);
    };
    const flag = () => {
      skip();
      if (d[i] === '0' || d[i] === '1') return d[i++] === '1';
      throw 0;
    };
    const hasNum = () => { skip(); NUM.lastIndex = i; return i < n && NUM.test(d); };
    let cx = 0, cy = 0, sx = 0, sy = 0, lcx = 0, lcy = 0, prev = '';
    try {
      while (true) {
        skip();
        if (i >= n) break;
        let cmd = d[i];
        if (/[a-zA-Z]/.test(cmd)) i++;
        else if (prev && hasNum()) cmd = prev === 'M' ? 'L' : prev === 'm' ? 'l' : prev;
        else break;
        const rel = cmd !== 'z' && cmd === cmd.toLowerCase();
        const C = cmd.toUpperCase();
        const ox = rel ? cx : 0, oy = rel ? cy : 0;
        switch (C) {
          case 'M': { const x = num() + ox, y = num() + oy; ops.push(['moveTo', x, y]); cx = sx = x; cy = sy = y; lcx = cx; lcy = cy; break; }
          case 'L': { const x = num() + ox, y = num() + oy; ops.push(['lineTo', x, y]); cx = x; cy = y; lcx = cx; lcy = cy; break; }
          case 'H': { const x = num() + ox; ops.push(['lineTo', x, cy]); cx = x; lcx = cx; lcy = cy; break; }
          case 'V': { const y = num() + oy; ops.push(['lineTo', cx, y]); cy = y; lcx = cx; lcy = cy; break; }
          case 'C': {
            const x1 = num() + ox, y1 = num() + oy, x2 = num() + ox, y2 = num() + oy, x = num() + ox, y = num() + oy;
            ops.push(['bezierCurveTo', x1, y1, x2, y2, x, y]); lcx = x2; lcy = y2; cx = x; cy = y; break;
          }
          case 'S': {
            const pc = /[CS]/i.test(prev);
            const x1 = pc ? 2 * cx - lcx : cx, y1 = pc ? 2 * cy - lcy : cy;
            const x2 = num() + ox, y2 = num() + oy, x = num() + ox, y = num() + oy;
            ops.push(['bezierCurveTo', x1, y1, x2, y2, x, y]); lcx = x2; lcy = y2; cx = x; cy = y; break;
          }
          case 'Q': {
            const x1 = num() + ox, y1 = num() + oy, x = num() + ox, y = num() + oy;
            ops.push(['quadraticCurveTo', x1, y1, x, y]); lcx = x1; lcy = y1; cx = x; cy = y; break;
          }
          case 'T': {
            const pq = /[QT]/i.test(prev);
            const x1 = pq ? 2 * cx - lcx : cx, y1 = pq ? 2 * cy - lcy : cy;
            const x = num() + ox, y = num() + oy;
            ops.push(['quadraticCurveTo', x1, y1, x, y]); lcx = x1; lcy = y1; cx = x; cy = y; break;
          }
          case 'A': {
            const rx = num(), ry = num(), phi = num(), fa = flag(), fs = flag(), x = num() + ox, y = num() + oy;
            svgArc(ops, cx, cy, rx, ry, phi, fa, fs, x, y); cx = x; cy = y; lcx = cx; lcy = cy; break;
          }
          case 'Z': ops.push(['closePath']); cx = sx; cy = sy; lcx = cx; lcy = cy; break;
          default: throw 0;
        }
        prev = cmd;
      }
    } catch (e) { if (e !== 0) throw e; /* 依 SVG 規格：遇到錯誤時保留錯誤之前的部分 */ }
    return ops;
  }
  function svgArc(ops, x1, y1, rx, ry, phiDeg, fa, fs, x2, y2) {
    if (x1 === x2 && y1 === y2) return;
    rx = Math.abs(rx); ry = Math.abs(ry);
    if (!rx || !ry) { ops.push(['lineTo', x2, y2]); return; }
    const phi = (phiDeg * Math.PI) / 180, c = Math.cos(phi), s = Math.sin(phi);
    const dx = (x1 - x2) / 2, dy = (y1 - y2) / 2;
    const xp = c * dx + s * dy, yp = -s * dx + c * dy;
    const lam = (xp * xp) / (rx * rx) + (yp * yp) / (ry * ry);
    if (lam > 1) { const k = Math.sqrt(lam); rx *= k; ry *= k; }
    const num = rx * rx * ry * ry - rx * rx * yp * yp - ry * ry * xp * xp;
    const den = rx * rx * yp * yp + ry * ry * xp * xp;
    let co = Math.sqrt(Math.max(0, num / den));
    if (fa === fs) co = -co;
    const cxp = (co * rx * yp) / ry, cyp = (-co * ry * xp) / rx;
    const cx = c * cxp - s * cyp + (x1 + x2) / 2, cy = s * cxp + c * cyp + (y1 + y2) / 2;
    const ang = (ux, uy, vx, vy) => Math.atan2(ux * vy - uy * vx, ux * vx + uy * vy);
    const ux = (xp - cxp) / rx, uy = (yp - cyp) / ry, vx = (-xp - cxp) / rx, vy = (-yp - cyp) / ry;
    const t1 = ang(1, 0, ux, uy);
    let dt = ang(ux, uy, vx, vy);
    if (!fs && dt > 0) dt -= TAU; else if (fs && dt < 0) dt += TAU;
    ops.push(['ellipse', cx, cy, rx, ry, phi, t1, t1 + dt, !fs]);
  }

  /* =====================================================================
   * Path2D（與原生 Path2D 相同介面；ctx.fill/stroke/clip 只接受此類別）
   * ===================================================================== */
  class Path2D {
    constructor(arg) {
      this._ops = [];
      if (arg instanceof Path2D) this._ops = arg._ops.slice();
      else if (typeof arg === 'string') this._ops = parseSVGPath(arg);
    }
    addPath(path, transform) {
      if (!(path instanceof Path2D)) throw new TypeError('addPath: argument is not a WebGL2D.Path2D');
      this._ops.push(['addPath', path._ops.slice(), transform ? toMatrix(transform) : null]);
    }
    closePath() { this._ops.push(['closePath']); }
    moveTo(x, y) { this._ops.push(['moveTo', +x, +y]); }
    lineTo(x, y) { this._ops.push(['lineTo', +x, +y]); }
    quadraticCurveTo(cpx, cpy, x, y) { this._ops.push(['quadraticCurveTo', +cpx, +cpy, +x, +y]); }
    bezierCurveTo(a, b, c, d, x, y) { this._ops.push(['bezierCurveTo', +a, +b, +c, +d, +x, +y]); }
    arc(x, y, r, sa, ea, ccw = false) {
      if (r < 0) throw domError('The radius provided is negative.', 'IndexSizeError');
      this._ops.push(['arc', +x, +y, +r, +sa, +ea, !!ccw]);
    }
    arcTo(x1, y1, x2, y2, r) {
      if (r < 0) throw domError('The radius provided is negative.', 'IndexSizeError');
      this._ops.push(['arcTo', +x1, +y1, +x2, +y2, +r]);
    }
    ellipse(x, y, rx, ry, rot, sa, ea, ccw = false) {
      if (rx < 0 || ry < 0) throw domError('The radius provided is negative.', 'IndexSizeError');
      this._ops.push(['ellipse', +x, +y, +rx, +ry, +rot, +sa, +ea, !!ccw]);
    }
    rect(x, y, w, h) { this._ops.push(['rect', +x, +y, +w, +h]); }
    roundRect(x, y, w, h, radii = 0) {
      new FlatPath(this._pr).roundRect([1, 0, 0, 1, 0, 0], 0, 0, 1, 1, radii); // 先驗證參數（會拋出例外）
      this._ops.push(['roundRect', +x, +y, +w, +h, Array.isArray(radii) ? radii.slice() : radii]);
    }
  }

  /* =====================================================================
   * 幾何收集器（每頂點 x, y, u, v）
   * ===================================================================== */
  class Geo {
    constructor(m) {
      this.v = []; this.m = m || null;
      this.x0 = Infinity; this.y0 = Infinity; this.x1 = -Infinity; this.y1 = -Infinity;
    }
    pt(x, y, u = 0, v = 0) {
      const m = this.m;
      if (m) { const tx = m[0] * x + m[2] * y + m[4]; y = m[1] * x + m[3] * y + m[5]; x = tx; }
      this.v.push(x, y, u, v);
      if (x < this.x0) this.x0 = x; if (x > this.x1) this.x1 = x;
      if (y < this.y0) this.y0 = y; if (y > this.y1) this.y1 = y;
    }
    tri(ax, ay, bx, by, cx, cy) { this.pt(ax, ay); this.pt(bx, by); this.pt(cx, cy); }
    get count() { return this.v.length / 4; }
    cover() {
      const x0 = Math.floor(this.x0) - 1, y0 = Math.floor(this.y0) - 1, x1 = Math.ceil(this.x1) + 1, y1 = Math.ceil(this.y1) + 1;
      return new Float32Array([x0, y0, 0, 0, x1, y0, 0, 0, x1, y1, 0, 0, x0, y0, 0, 0, x1, y1, 0, 0, x0, y1, 0, 0]);
    }
  }

  function fanGeometry(subs) {
    const g = new Geo(null);
    for (const s of subs) {
      const p = s.pts, n = p.length / 2;
      if (n < 3) continue;
      for (let i = 1; i < n - 1; i++) g.tri(p[0], p[1], p[2 * i], p[2 * i + 1], p[2 * i + 2], p[2 * i + 3]);
    }
    return g;
  }

  /* =====================================================================
   * 描邊幾何（使用者座標產生，再以 CTM 轉到裝置座標）
   * ===================================================================== */
  function arcFan(g, cx, cy, r, a0, sweep, segAng) {
    const n = Math.max(1, Math.ceil(Math.abs(sweep) / segAng));
    let px = cx + r * Math.cos(a0), py = cy + r * Math.sin(a0);
    for (let i = 1; i <= n; i++) {
      const a = a0 + (sweep * i) / n, qx = cx + r * Math.cos(a), qy = cy + r * Math.sin(a);
      g.tri(cx, cy, px, py, qx, qy); px = qx; py = qy;
    }
  }
  function strokeJoin(g, px, py, d0x, d0y, d1x, d1y, hw, st, segAng) {
    const cr = d0x * d1y - d0y * d1x, dt = d0x * d1x + d0y * d1y;
    if (Math.abs(cr) < 1e-12 && dt > 0) return;
    if (st.lineJoin === 'round') {
      if (Math.abs(cr) < 1e-12) { arcFan(g, px, py, hw, 0, TAU, segAng); return; }
      const s = cr > 0 ? -1 : 1;
      const a0 = Math.atan2(d0x * s, -d0y * s), a1 = Math.atan2(d1x * s, -d1y * s);
      let d = a1 - a0;
      while (d > Math.PI) d -= TAU;
      while (d < -Math.PI) d += TAU;
      arcFan(g, px, py, hw, a0, d, segAng);
      return;
    }
    const s = cr > 0 ? -1 : 1;
    const n0x = -d0y * s * hw, n0y = d0x * s * hw, n1x = -d1y * s * hw, n1y = d1x * s * hw;
    if (st.lineJoin === 'miter') {
      const bx = n0x + n1x, by = n0y + n1y, bb = bx * bx + by * by;
      if (bb > 1e-24 && (2 * hw) / Math.sqrt(bb) <= st.miterLimit) {
        const k = (2 * hw * hw) / bb, tx = px + bx * k, ty = py + by * k;
        g.tri(px, py, px + n0x, py + n0y, tx, ty);
        g.tri(px, py, tx, ty, px + n1x, py + n1y);
        return;
      }
    }
    g.tri(px, py, px + n0x, py + n0y, px + n1x, py + n1y);
  }
  function strokeCap(g, px, py, dx, dy, hw, st, segAng, atEnd) {
    if (st.lineCap === 'round') arcFan(g, px, py, hw, 0, TAU, segAng);
    else if (st.lineCap === 'square') {
      const s = atEnd ? 1 : -1, ex = dx * hw * s, ey = dy * hw * s, nx = -dy * hw, ny = dx * hw;
      g.tri(px + nx, py + ny, px - nx, py - ny, px - nx + ex, py - ny + ey);
      g.tri(px + nx, py + ny, px - nx + ex, py - ny + ey, px + nx + ex, py + ny + ey);
    }
  }
  function strokePolyline(g, pts, closed, hw, st, segAng, eps) {
    const q = [];
    for (let i = 0; i < pts.length; i += 2) {
      const n = q.length;
      if (n && Math.abs(q[n - 2] - pts[i]) <= eps && Math.abs(q[n - 1] - pts[i + 1]) <= eps) continue;
      q.push(pts[i], pts[i + 1]);
    }
    if (closed && q.length >= 4 && Math.abs(q[0] - q[q.length - 2]) <= eps && Math.abs(q[1] - q[q.length - 1]) <= eps) q.length -= 2;
    const n = q.length / 2;
    if (n === 1) {
      if (closed) return;
      const [x, y] = q;
      if (st.lineCap === 'round') arcFan(g, x, y, hw, 0, TAU, segAng);
      else if (st.lineCap === 'square') { g.tri(x - hw, y - hw, x + hw, y - hw, x + hw, y + hw); g.tri(x - hw, y - hw, x + hw, y + hw, x - hw, y + hw); }
      return;
    }
    const S = closed ? n : n - 1, dirs = new Array(S * 2);
    for (let k = 0; k < S; k++) {
      const a = k, b = (k + 1) % n;
      const ax = q[2 * a], ay = q[2 * a + 1], bx = q[2 * b], by = q[2 * b + 1];
      let dx = bx - ax, dy = by - ay; const L = Math.hypot(dx, dy); dx /= L; dy /= L;
      dirs[2 * k] = dx; dirs[2 * k + 1] = dy;
      const nx = -dy * hw, ny = dx * hw;
      g.tri(ax + nx, ay + ny, bx + nx, by + ny, bx - nx, by - ny);
      g.tri(ax + nx, ay + ny, bx - nx, by - ny, ax - nx, ay - ny);
    }
    const j0 = closed ? 0 : 1, j1 = closed ? n : n - 1;
    for (let i = j0; i < j1; i++) {
      const pk = (i - 1 + S) % S;
      strokeJoin(g, q[2 * i], q[2 * i + 1], dirs[2 * pk], dirs[2 * pk + 1], dirs[2 * i], dirs[2 * i + 1], hw, st, segAng);
    }
    if (!closed) {
      strokeCap(g, q[0], q[1], dirs[0], dirs[1], hw, st, segAng, false);
      strokeCap(g, q[2 * n - 2], q[2 * n - 1], dirs[2 * S - 2], dirs[2 * S - 1], hw, st, segAng, true);
    }
  }
  function dashPolyline(pts, closed, dash, offset) {
    const len = dash.length;
    let total = 0;
    for (const d of dash) total += d;
    if (!(total > 0)) return [pts];
    const out = [];
    let idx = 0, off = ((offset % total) + total) % total, guard = 0;
    while (off >= dash[idx] && guard++ < 1e5) { off -= dash[idx]; idx = (idx + 1) % len; }
    let rem = dash[idx] - off, on = idx % 2 === 0;
    let cur = on ? [pts[0], pts[1]] : null;
    const n = pts.length / 2, segs = closed ? n : n - 1;
    guard = 0;
    for (let k = 0; k < segs; k++) {
      const ax = pts[2 * k], ay = pts[2 * k + 1], b = (k + 1) % n, bx = pts[2 * b], by = pts[2 * b + 1];
      const L = Math.hypot(bx - ax, by - ay);
      let pos = 0;
      while (L - pos > rem && guard++ < 1e6) {
        pos += rem;
        const t = pos / L, x = ax + (bx - ax) * t, y = ay + (by - ay) * t;
        if (on) { cur.push(x, y); out.push(cur); cur = null; } else cur = [x, y];
        on = !on; idx = (idx + 1) % len; rem = dash[idx];
      }
      rem -= L - pos;
      if (on) cur.push(bx, by);
    }
    if (cur && cur.length >= 4) out.push(cur);
    return out;
  }
  function buildStroke(subs, m, st, pr = 1) {
    const im = inv(m);
    if (!im) return null;
    const g = new Geo(m);
    const hw = st.lineWidth / 2, sc = maxScale(m) * pr, rpx = hw * sc;
    const segAng = Math.max(0.02, rpx > 0.25 ? 2 * Math.acos(Math.max(-1, 1 - 0.25 / rpx)) : Math.PI / 2);
    const eps = 1e-7 / (sc || 1);
    const dash = st.lineDash.length ? st.lineDash : null;
    for (const s of subs) {
      const p = s.pts;
      if (p.length < 4) continue;
      const up = new Array(p.length);
      for (let i = 0; i < p.length; i += 2) {
        up[i] = im[0] * p[i] + im[2] * p[i + 1] + im[4];
        up[i + 1] = im[1] * p[i] + im[3] * p[i + 1] + im[5];
      }
      if (dash) {
        for (const piece of dashPolyline(up, s.closed, dash, st.lineDashOffset)) strokePolyline(g, piece, false, hw, st, segAng, eps);
      } else strokePolyline(g, up, s.closed, hw, st, segAng, eps);
    }
    return g;
  }

  /* =====================================================================
   * 命中測試
   * ===================================================================== */
  function windingAt(subs, x, y) {
    let w = 0;
    for (const s of subs) {
      const p = s.pts, n = p.length / 2;
      if (n < 2) continue;
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n, x1 = p[2 * i], y1 = p[2 * i + 1], x2 = p[2 * j], y2 = p[2 * j + 1];
        const c = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1);
        if (y1 <= y) { if (y2 > y && c > 0) w++; } else if (y2 <= y && c < 0) w--;
      }
    }
    return w;
  }
  function pointInTris(v, x, y) {
    for (let i = 0; i < v.length; i += 12) {
      const ax = v[i], ay = v[i + 1], bx = v[i + 4], by = v[i + 5], cx = v[i + 8], cy = v[i + 9];
      const d1 = (x - bx) * (ay - by) - (ax - bx) * (y - by);
      const d2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy);
      const d3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay);
      const neg = d1 < 0 || d2 < 0 || d3 < 0, pos = d1 > 0 || d2 > 0 || d3 > 0;
      if (!(neg && pos)) return true;
    }
    return false;
  }

  /* =====================================================================
   * 漸層 / 圖樣
   * ===================================================================== */
  const RAMP = 512;
  class CanvasGradient {
    constructor(ctx, type, p) { this._ctx = ctx; this._type = type; this._p = p; this._stops = []; this._tex = null; this._dirty = true; }
    addColorStop(offset, color) {
      offset = +offset;
      if (!(offset >= 0 && offset <= 1)) throw domError('The provided offset is outside [0, 1].', 'IndexSizeError');
      const c = parseColor(String(color));
      if (!c) throw domError(`The value provided ('${color}') could not be parsed as a color.`, 'SyntaxError');
      this._stops.push({ o: offset, c });
      this._stops.sort((a, b) => a.o - b.o); // Array.prototype.sort 為穩定排序：同 offset 依加入順序
      this._dirty = true;
    }
    _texture(gl) {
      if (!this._tex) this._tex = gl.createTexture();
      if (this._dirty) {
        const st = this._stops, data = new Uint8Array(RAMP * 4);
        for (let i = 0; i < RAMP; i++) {
          const t = i / (RAMP - 1);
          let c;
          if (t <= st[0].o) c = st[0].c;
          else if (t >= st[st.length - 1].o) c = st[st.length - 1].c;
          else {
            let k = 0;
            while (k < st.length - 2 && st[k + 1].o <= t) k++;
            const a = st[k], b = st[k + 1], f = b.o > a.o ? (t - a.o) / (b.o - a.o) : 1;
            // 與瀏覽器的 Canvas 2D 一致：在非預乘 RGBA 空間內插，再預乘存入紋理
            c = a.c.map((v, j) => v + (b.c[j] - v) * f);
          }
          data.set([c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3] * 255].map(Math.round), i * 4);
        }
        gl.bindTexture(gl.TEXTURE_2D, this._tex);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, RAMP, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        this._dirty = false;
      }
      return this._tex;
    }
  }
  class CanvasPattern {
    constructor(tex, w, h, rep, flipY, tw = w, th = h) {
      this._tex = tex; this._w = w; this._h = h; this._tw = tw; this._th = th;
      this._rep = rep; this._flipY = flipY; this._m = [1, 0, 0, 1, 0, 0];
    }
    setTransform(m) { this._m = toMatrix(m); }
  }

  /* =====================================================================
   * Shaders
   * ===================================================================== */
  const VS = `
attribute vec2 a_pos;
attribute vec2 a_uv;
uniform vec2 u_res;
uniform mat3 u_paintInv;
varying vec2 v_paint;
varying vec2 v_uv;
void main() {
  vec2 c = a_pos / u_res * 2.0 - 1.0;
  gl_Position = vec4(c.x, -c.y, 0.0, 1.0);
  v_paint = (u_paintInv * vec3(a_pos, 1.0)).xy;
  v_uv = a_uv;
}`;
  const FS = `
precision highp float;
varying vec2 v_paint;
varying vec2 v_uv;
uniform int u_type;          // 0 純色 1 線性 2 放射 3 圓錐 4 圖樣 5 影像
uniform vec4 u_color;        // 預乘 alpha
uniform vec4 u_g0;
uniform vec4 u_g1;
uniform sampler2D u_ramp;
uniform sampler2D u_tex;
uniform sampler2D u_mask;
uniform int u_useMask;
uniform float u_alpha;
uniform vec2 u_patSize;
uniform vec2 u_repeat;
uniform float u_patFlip;
uniform int u_quality;       // 0 最近鄰 1 low 2 medium 3 high
uniform vec2 u_texSize;      // 紋理尺寸（texel）
uniform vec2 u_dx;           // 往右移動一個實體像素時 uv 的變化量
uniform vec2 u_dy;           // 往下移動一個實體像素時 uv 的變化量
const float TAU = 6.28318530717959;
// 單點取樣：圖樣在此處理重複 / 不重複 / 翻轉（每個取樣點各自折返，不會有接縫）
vec4 tap(vec2 uv) {
  if (u_type == 4) {
    if (u_repeat.x > 0.5) uv.x = fract(uv.x); else if (uv.x < 0.0 || uv.x > 1.0) return vec4(0.0);
    if (u_repeat.y > 0.5) uv.y = fract(uv.y); else if (uv.y < 0.0 || uv.y > 1.0) return vec4(0.0);
    uv.y = mix(uv.y, 1.0 - uv.y, u_patFlip);
  }
  return texture2D(u_tex, uv);
}
// Mitchell–Netravali（B = C = 1/3）
float mitchell(float x) {
  x = abs(x);
  if (x < 1.0) return (7.0 * x * x * x - 12.0 * x * x + 16.0 / 3.0) / 6.0;
  if (x < 2.0) return (-7.0 / 3.0 * x * x * x + 12.0 * x * x - 20.0 * x + 32.0 / 3.0) / 6.0;
  return 0.0;
}
vec4 bicubic(vec2 uv) {
  vec2 p = uv * u_texSize - 0.5, b = floor(p), f = p - b;
  vec4 sum = vec4(0.0);
  for (int j = -1; j <= 2; j++) {
    float wy = mitchell(float(j) - f.y);
    for (int i = -1; i <= 2; i++) {
      sum += tap((b + vec2(float(i), float(j)) + 0.5) / u_texSize) * (mitchell(float(i) - f.x) * wy);
    }
  }
  sum = clamp(sum, 0.0, 1.0);
  return vec4(min(sum.rgb, vec3(sum.a)), sum.a);   // 保持合法的預乘顏色
}
vec4 sampleTex(vec2 uv) {
  if (u_quality <= 1) return tap(uv);                       // 最近鄰 / 雙線性
  float fx = length(u_dx * u_texSize), fy = length(u_dy * u_texSize);   // 每個像素涵蓋幾個 texel
  if (max(fx, fy) <= 1.0) return u_quality == 3 ? bicubic(uv) : tap(uv);
  // 縮小：在像素涵蓋範圍內超取樣（每個取樣點本身是雙線性）
  //   medium：box 濾波，範圍 = 1 個輸出像素（效果相當於 mipmap）
  //   high  ：Mitchell 濾波，範圍 = 4 個輸出像素（半徑 2），兼顧銳利與抗鋸齒，與放大時使用同一個核
  float k = u_quality == 3 ? 4.0 : 1.0;
  int nx = int(clamp(ceil(fx * k), 1.0, 12.0)), ny = int(clamp(ceil(fy * k), 1.0, 12.0));
  vec4 sum = vec4(0.0);
  float ws = 0.0;
  for (int j = 0; j < 12; j++) {
    if (j >= ny) break;
    for (int i = 0; i < 12; i++) {
      if (i >= nx) break;
      vec2 o = ((vec2(float(i), float(j)) + 0.5) / vec2(float(nx), float(ny)) - 0.5) * k;   // 以輸出像素為單位
      float w = u_quality == 3 ? mitchell(o.x) * mitchell(o.y) : 1.0;
      sum += tap(uv + u_dx * o.x + u_dy * o.y) * w;
      ws += w;
    }
  }
  sum /= ws;
  sum = clamp(sum, 0.0, 1.0);
  return vec4(min(sum.rgb, vec3(sum.a)), sum.a);
}
vec4 ramp(float t) { return texture2D(u_ramp, vec2(clamp(t, 0.0, 1.0) * ${(RAMP - 1) / RAMP} + ${0.5 / RAMP}, 0.5)); }
void main() {
  vec4 c = vec4(0.0);
  if (u_type == 0) {
    c = u_color;
  } else if (u_type == 1) {
    vec2 d = u_g0.zw - u_g0.xy;
    c = ramp(dot(v_paint - u_g0.xy, d) / dot(d, d));
  } else if (u_type == 2) {
    vec2 cd = u_g1.xy - u_g0.xy;
    vec2 pd = v_paint - u_g0.xy;
    float r0 = u_g0.z;
    float dr = u_g1.z - r0;
    float a = dot(cd, cd) - dr * dr;
    float b = dot(pd, cd) + r0 * dr;
    float k = dot(pd, pd) - r0 * r0;
    float t = 0.0;
    bool ok = true;
    if (abs(a) < 1e-6) {
      if (abs(b) < 1e-9) ok = false;
      else { t = k / (2.0 * b); ok = r0 + t * dr >= 0.0; }
    } else {
      float disc = b * b - a * k;
      if (disc < 0.0) ok = false;
      else {
        float sq = sqrt(disc);
        float t1 = (b + sq) / a;
        float t2 = (b - sq) / a;
        float hi = max(t1, t2);
        float lo = min(t1, t2);
        if (r0 + hi * dr >= 0.0) t = hi;
        else if (r0 + lo * dr >= 0.0) t = lo;
        else ok = false;
      }
    }
    if (ok) c = ramp(t);
  } else if (u_type == 3) {
    vec2 d = v_paint - u_g0.xy;
    c = ramp(fract((atan(d.y, d.x) - u_g0.z) / TAU));
  } else if (u_type == 4) {
    c = sampleTex(v_paint / u_patSize);
  } else {
    c = sampleTex(v_uv);
  }
  if (u_useMask == 1) c *= texture2D(u_mask, v_uv).a;
  gl_FragColor = c * u_alpha;
}`;

  /* =====================================================================
   * 狀態
   * ===================================================================== */
  /* =====================================================================
   * 濾鏡（filter / 陰影 / 自訂 shader）
   *   繪圖先畫進離屏圖層 → 依序執行濾鏡 pass（ping-pong 紋理）→ 依裁切、透明度、合成模式貼回
   *   濾鏡 pass 的紋理使用 GL 方向（uv 的 y 向上）
   * ===================================================================== */
  const FX_VS = `
attribute vec2 a_pos;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() { v_uv = a_uv; gl_Position = vec4(a_pos, 0.0, 1.0); }`;
  const FX_QUAD = new Float32Array([-1, -1, 0, 0, 1, -1, 1, 0, 1, 1, 1, 1, -1, -1, 0, 0, 1, 1, 1, 1, -1, 1, 0, 1]);
  const FX_HEAD = 'precision highp float;\nvarying vec2 v_uv;\nuniform sampler2D u_src;\n';
  const FX_INSIDE = 'uniform vec4 u_box;\nbool inside(vec2 uv) { return uv.x >= u_box.x && uv.x <= u_box.z && uv.y >= u_box.y && uv.y <= u_box.w; }\n';
  const FX_FS = {
    // 可分離高斯模糊：u_dir 為單一 texel 的方向向量，u_sigma 以實體像素計
    blur: FX_HEAD + FX_INSIDE + `
uniform vec2 u_dir;
uniform float u_sigma;
void main() {
  float st = max(1.0, ceil(3.0 * u_sigma / 24.0));
  vec4 sum = vec4(0.0);
  float ws = 0.0;
  for (int i = -24; i <= 24; i++) {
    float x = float(i) * st;
    float w = exp(-0.5 * x * x / (u_sigma * u_sigma));
    vec2 uv = v_uv + u_dir * x;
    if (inside(uv)) sum += texture2D(u_src, uv) * w;   // 畫布外視為透明
    ws += w;
  }
  gl_FragColor = sum / ws;
}`,
    // 色彩矩陣（作用在非預乘的 RGBA 上）
    matrix: FX_HEAD + `
uniform mat4 u_m;
uniform vec4 u_off;
void main() {
  vec4 c = texture2D(u_src, v_uv);
  vec4 u = c.a > 0.0 ? vec4(c.rgb / c.a, c.a) : vec4(0.0);
  vec4 r = clamp(u_m * u + u_off, 0.0, 1.0);
  gl_FragColor = vec4(r.rgb * r.a, r.a);
}`,
    // 陰影：取來源的 alpha、位移、上色
    tint: FX_HEAD + FX_INSIDE + `
uniform vec4 u_color;
uniform vec2 u_off;
void main() {
  vec2 uv = v_uv - u_off;
  gl_FragColor = inside(uv) ? u_color * texture2D(u_src, uv).a : vec4(0.0);
}`,
    // 來源疊在陰影上（source-over）
    over: FX_HEAD + `
uniform sampler2D u_under;
void main() {
  vec4 s = texture2D(u_src, v_uv);
  gl_FragColor = s + texture2D(u_under, v_uv) * (1.0 - s.a);
}`,
  };
  // W3C Compositing and Blending Level 1：所有混合模式皆以 source-over 合成
  //   Cr = (1 - αb)·Cs + (1 - αs)·Cb + αs·αb·B(cs, cb)（C 為預乘、c 為非預乘）
  FX_FS.blend = FX_HEAD + `
uniform sampler2D u_dst;
uniform int u_mode;
uniform float u_alpha;
float lum(vec3 c) { return dot(c, vec3(0.3, 0.59, 0.11)); }
vec3 clipColor(vec3 c) {
  float l = lum(c), n = min(min(c.r, c.g), c.b), x = max(max(c.r, c.g), c.b);
  if (n < 0.0) c = l + (c - l) * l / (l - n);
  if (x > 1.0) c = l + (c - l) * (1.0 - l) / (x - l);
  return c;
}
vec3 setLum(vec3 c, float l) { return clipColor(c + (l - lum(c))); }
float sat(vec3 c) { return max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b); }
vec3 setSat(vec3 c, float s) {
  float mx = max(max(c.r, c.g), c.b), mn = min(min(c.r, c.g), c.b);
  return mx > mn ? (c - mn) * s / (mx - mn) : vec3(0.0);
}
float hardLight(float s, float b) { return s <= 0.5 ? b * 2.0 * s : b + (2.0 * s - 1.0) - b * (2.0 * s - 1.0); }
float sep(float s, float b) {
  if (u_mode == 1) return s * b;                                   // multiply
  if (u_mode == 2) return s + b - s * b;                           // screen
  if (u_mode == 3) return hardLight(b, s);                         // overlay
  if (u_mode == 4) return min(s, b);                               // darken
  if (u_mode == 5) return max(s, b);                               // lighten
  if (u_mode == 6) return b == 0.0 ? 0.0 : (s >= 1.0 ? 1.0 : min(1.0, b / (1.0 - s)));          // color-dodge
  if (u_mode == 7) return b >= 1.0 ? 1.0 : (s <= 0.0 ? 0.0 : 1.0 - min(1.0, (1.0 - b) / s));   // color-burn
  if (u_mode == 8) return hardLight(s, b);                         // hard-light
  if (u_mode == 9) {                                               // soft-light
    if (s <= 0.5) return b - (1.0 - 2.0 * s) * b * (1.0 - b);
    float d = b <= 0.25 ? ((16.0 * b - 12.0) * b + 4.0) * b : sqrt(b);
    return b + (2.0 * s - 1.0) * (d - b);
  }
  if (u_mode == 10) return abs(b - s);                             // difference
  return b + s - 2.0 * b * s;                                      // exclusion (11)
}
vec3 blendFn(vec3 s, vec3 b) {
  if (u_mode == 12) return setLum(setSat(s, sat(b)), lum(b));      // hue
  if (u_mode == 13) return setLum(setSat(b, sat(s)), lum(b));      // saturation
  if (u_mode == 14) return setLum(s, lum(b));                      // color
  if (u_mode == 15) return setLum(b, lum(s));                      // luminosity
  return vec3(sep(s.r, b.r), sep(s.g, b.g), sep(s.b, b.b));
}
void main() {
  vec4 S = texture2D(u_src, v_uv) * u_alpha;
  vec4 D = texture2D(u_dst, v_uv);
  float as = S.a, ab = D.a;
  vec3 cs = as > 0.0 ? S.rgb / as : vec3(0.0);
  vec3 cb = ab > 0.0 ? D.rgb / ab : vec3(0.0);
  vec3 B = clamp(blendFn(clamp(cs, 0.0, 1.0), clamp(cb, 0.0, 1.0)), 0.0, 1.0);
  gl_FragColor = vec4((1.0 - ab) * S.rgb + (1.0 - as) * D.rgb + as * ab * B, as + ab * (1.0 - as));
}`;
  const BLEND_MODES = { multiply: 1, overlay: 3, darken: 4, lighten: 5, 'color-dodge': 6, 'color-burn': 7,
    'hard-light': 8, 'soft-light': 9, difference: 10, exclusion: 11, hue: 12, saturation: 13, color: 14, luminosity: 15 };
  // screen 可以用 blendFunc 精確表達，走快速路徑

  // 自訂 shader 的前置程式碼
  const FX_USER_PRELUDE = FX_HEAD + `uniform vec2 u_resolution;   // 實體像素
uniform vec2 u_size;         // CSS 像素
uniform float u_pixelRatio;
vec4 texel(vec2 uv) { return texture2D(u_src, uv); }
vec4 unpremultiply(vec4 c) { return c.a > 0.0 ? vec4(c.rgb / c.a, c.a) : vec4(0.0); }
vec4 premultiply(vec4 c) { return vec4(c.rgb * c.a, c.a); }
vec2 toCSS(vec2 uv) { return vec2(uv.x, 1.0 - uv.y) * u_size; }
vec2 fromCSS(vec2 p) { return vec2(p.x / u_size.x, 1.0 - p.y / u_size.y); }
`;

  function compileProgram(gl, vs, fs, label) {
    const sh = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(s); gl.deleteShader(s);
        throw new SyntaxError(`${label} 編譯失敗：\n${log}`);
      }
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, 'a_pos'); gl.bindAttribLocation(p, 1, 'a_uv');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new SyntaxError(`${label} 連結失敗：\n${gl.getProgramInfoLog(p)}`);
    const info = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const u = gl.getActiveUniform(p, i);
      info[u.name.replace(/\[0\]$/, '')] = { loc: gl.getUniformLocation(p, u.name), type: u.type, size: u.size };
    }
    return { p, info, u: (name) => (info[name] ? info[name].loc : null) };
  }

  /* ---------- CSS filter 解析 ---------- */
  // 4×4 色彩矩陣（列優先）+ 位移 → GL 欄優先陣列
  function cmOp(rows3, off3 = [0, 0, 0], alpha = 1) {
    const R = [[...rows3[0], 0], [...rows3[1], 0], [...rows3[2], 0], [0, 0, 0, alpha]];
    const m = [];
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) m.push(R[r][c]);
    return { t: 'matrix', m, off: [...off3, 0] };
  }
  const DIAG = (a) => [[a, 0, 0], [0, a, 0], [0, 0, a]];
  function colorMatrixFor(name, a) {
    switch (name) {
      case 'brightness': return cmOp(DIAG(a));
      case 'contrast': return cmOp(DIAG(a), [(1 - a) / 2, (1 - a) / 2, (1 - a) / 2]);
      case 'invert': { const i = Math.min(a, 1); return cmOp(DIAG(1 - 2 * i), [i, i, i]); }
      case 'opacity': return cmOp(DIAG(1), undefined, Math.min(a, 1));
      case 'grayscale': { const s = 1 - Math.min(a, 1); return cmOp([
        [0.2126 + 0.7874 * s, 0.7152 - 0.7152 * s, 0.0722 - 0.0722 * s],
        [0.2126 - 0.2126 * s, 0.7152 + 0.2848 * s, 0.0722 - 0.0722 * s],
        [0.2126 - 0.2126 * s, 0.7152 - 0.7152 * s, 0.0722 + 0.9278 * s]]); }
      case 'sepia': { const s = 1 - Math.min(a, 1); return cmOp([
        [0.393 + 0.607 * s, 0.769 - 0.769 * s, 0.189 - 0.189 * s],
        [0.349 - 0.349 * s, 0.686 + 0.314 * s, 0.168 - 0.168 * s],
        [0.272 - 0.272 * s, 0.534 - 0.534 * s, 0.131 + 0.869 * s]]); }
      case 'saturate': return cmOp([
        [0.213 + 0.787 * a, 0.715 - 0.715 * a, 0.072 - 0.072 * a],
        [0.213 - 0.213 * a, 0.715 + 0.285 * a, 0.072 - 0.072 * a],
        [0.213 - 0.213 * a, 0.715 - 0.715 * a, 0.072 + 0.928 * a]]);
      case 'hue-rotate': { const c = Math.cos(a), s = Math.sin(a); return cmOp([
        [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
        [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283],
        [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072]]); }
    }
    return null;
  }
  function cssLength(s) {
    if (s === '0') return 0;
    const m = /^([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)px$/i.exec(s);
    return m ? parseFloat(m[1]) : NaN;
  }
  function cssAngle(s) {
    if (s === '0') return 0;
    const m = /^([-+]?(?:\d+\.?\d*|\.\d+))(deg|rad|grad|turn)$/i.exec(s);
    if (!m) return NaN;
    const v = parseFloat(m[1]), u = m[2].toLowerCase();
    return u === 'deg' ? (v * Math.PI) / 180 : u === 'rad' ? v : u === 'grad' ? (v * Math.PI) / 200 : v * TAU;
  }
  function cssAmount(s, def) {
    if (!s) return def;
    const v = s.endsWith('%') ? parseFloat(s) / 100 : Number(s);
    return Number.isFinite(v) && v >= 0 ? v : NaN;
  }
  function parseFilterFn(name, arg) {
    name = name.toLowerCase();
    if (name === 'blur') { const v = arg ? cssLength(arg) : 0; return v >= 0 ? { t: 'blur', sigma: v } : null; }
    if (name === 'hue-rotate') { const a = arg ? cssAngle(arg) : 0; return Number.isFinite(a) ? colorMatrixFor(name, a) : null; }
    if (name === 'drop-shadow') {
      const toks = arg.match(/[^\s(]+(\([^)]*\))?/g) || [];
      const lens = [], rest = [];
      for (const t of toks) { const l = cssLength(t); if (Number.isFinite(l)) lens.push(l); else rest.push(t); }
      if (lens.length < 2 || lens.length > 3 || (lens[2] || 0) < 0) return null;
      const color = rest.length ? parseColor(rest.join(' ')) : [0, 0, 0, 1];
      if (!color) return null;
      return { t: 'shadow', ox: lens[0], oy: lens[1], sigma: (lens[2] || 0) / 2, color };
    }
    if (['brightness', 'contrast', 'grayscale', 'sepia', 'saturate', 'invert', 'opacity'].includes(name)) {
      const a = cssAmount(arg, 1);
      return Number.isFinite(a) ? colorMatrixFor(name, a) : null;
    }
    return null;
  }
  /** 解析 CSS filter 字串 → 濾鏡步驟陣列；'none' → []；無效 → null */
  function parseCSSFilter(str) {
    str = String(str).trim();
    if (str === 'none' || str === '') return [];
    const re = /\s*([a-zA-Z-]+)\(\s*((?:[^()]|\([^()]*\))*?)\s*\)\s*/y;
    const out = [];
    re.lastIndex = 0;
    while (re.lastIndex < str.length) {
      const m = re.exec(str);
      if (!m) return null;
      const f = parseFilterFn(m[1], m[2]);
      if (!f) return null;
      out.push(f);
    }
    return out;
  }

  /* ---------- 自訂 shader 濾鏡 ---------- */
  class ShaderFilter {
    constructor(ctx, source, uniforms) {
      this._ctx = ctx;
      this.source = String(source);
      this._prog = compileProgram(ctx.gl, FX_VS, FX_USER_PRELUDE + '#line 1\n' + this.source + '\nvoid main() { gl_FragColor = effect(v_uv); }\n', 'ShaderFilter');
      this.uniforms = Object.assign({}, uniforms);
      this._texs = {};
    }
    /** 設定 uniform：number / 陣列 / boolean / CSS 顏色字串（vec3、vec4）/ 影像來源（sampler2D） */
    setUniform(name, value) { this.uniforms[name] = value; return this; }
    get uniformNames() { return Object.keys(this._prog.info).filter((n) => !/^u_(src|resolution|size|pixelRatio)$/.test(n)); }
    dispose() {
      const gl = this._ctx.gl;
      gl.deleteProgram(this._prog.p);
      for (const k in this._texs) gl.deleteTexture(this._texs[k].tex);
      this._texs = {};
    }
    _apply(gl, unitStart) {
      let unit = unitStart;
      for (const [name, v] of Object.entries(this.uniforms)) {
        const u = this._prog.info[name];
        if (!u || v === undefined || v === null) continue;
        const T = u.type;
        const f32 = (x) => {
          if (typeof x === 'string') { const c = parseColor(x); if (!c) return null; return new Float32Array(T === gl.FLOAT_VEC3 ? [c[0] / 255, c[1] / 255, c[2] / 255] : [c[0] / 255, c[1] / 255, c[2] / 255, c[3]]); }
          return new Float32Array(typeof x === 'number' || typeof x === 'boolean' ? [+x] : Array.from(x, Number));
        };
        const i32 = (x) => new Int32Array(typeof x === 'number' || typeof x === 'boolean' ? [+x] : Array.from(x, Number));
        switch (T) {
          case gl.FLOAT: gl.uniform1fv(u.loc, f32(v)); break;
          case gl.FLOAT_VEC2: gl.uniform2fv(u.loc, f32(v)); break;
          case gl.FLOAT_VEC3: { const a = f32(v); if (a) gl.uniform3fv(u.loc, a); break; }
          case gl.FLOAT_VEC4: { const a = f32(v); if (a) gl.uniform4fv(u.loc, a); break; }
          case gl.INT: case gl.BOOL: gl.uniform1iv(u.loc, i32(v)); break;
          case gl.INT_VEC2: case gl.BOOL_VEC2: gl.uniform2iv(u.loc, i32(v)); break;
          case gl.INT_VEC3: case gl.BOOL_VEC3: gl.uniform3iv(u.loc, i32(v)); break;
          case gl.INT_VEC4: case gl.BOOL_VEC4: gl.uniform4iv(u.loc, i32(v)); break;
          case gl.FLOAT_MAT2: gl.uniformMatrix2fv(u.loc, false, f32(v)); break;
          case gl.FLOAT_MAT3: gl.uniformMatrix3fv(u.loc, false, f32(v)); break;
          case gl.FLOAT_MAT4: gl.uniformMatrix4fv(u.loc, false, f32(v)); break;
          case gl.SAMPLER_2D: {
            let e = this._texs[name];
            if (!e) e = this._texs[name] = { tex: gl.createTexture(), src: null };
            gl.activeTexture(gl.TEXTURE0 + unit);
            gl.bindTexture(gl.TEXTURE_2D, e.tex);
            const el = resolveSource(v).el;
            if (e.src !== el || !isStatic(el)) {
              gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
              gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true); // 與濾鏡紋理同方向（y 向上）
              gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, el);
              gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
              e.src = el;
            }
            gl.uniform1i(u.loc, unit++);
            break;
          }
        }
      }
      gl.activeTexture(gl.TEXTURE0);
      return unit;
    }
  }
  /** filter 值（字串 / ShaderFilter / 兩者混合的陣列）→ 步驟陣列；無效 → null */
  function buildFilterChain(v) {
    if (v instanceof ShaderFilter) return [{ t: 'custom', f: v }];
    if (Array.isArray(v)) {
      const out = [];
      for (const x of v) { const c = buildFilterChain(x); if (!c) return null; out.push(...c); }
      return out;
    }
    if (typeof v === 'string') return parseCSSFilter(v);
    return null;
  }

  const COMPOSITE = new Set(['source-over', 'source-in', 'source-out', 'source-atop', 'destination-over', 'destination-in',
    'destination-out', 'destination-atop', 'lighter', 'copy', 'xor', 'multiply', 'screen', 'overlay', 'darken', 'lighten',
    'color-dodge', 'color-burn', 'hard-light', 'soft-light', 'difference', 'exclusion', 'hue', 'saturation', 'color', 'luminosity']);
  const UNBOUNDED = new Set(['source-in', 'source-out', 'destination-in', 'destination-atop', 'copy']);

  function defaultState() {
    return {
      m: [1, 0, 0, 1, 0, 0],
      fill: { color: [0, 0, 0, 1] }, stroke: { color: [0, 0, 0, 1] },
      globalAlpha: 1, globalCompositeOperation: 'source-over',
      lineWidth: 1, lineCap: 'butt', lineJoin: 'miter', miterLimit: 10, lineDash: [], lineDashOffset: 0,
      font: '10px sans-serif', textAlign: 'start', textBaseline: 'alphabetic', direction: 'inherit',
      letterSpacing: '0px', wordSpacing: '0px', fontKerning: 'auto', fontStretch: 'normal', fontVariantCaps: 'normal', textRendering: 'auto',
      imageSmoothingEnabled: true, imageSmoothingQuality: 'low',
      shadowBlur: 0, shadowColor: 'rgba(0, 0, 0, 0)', shadowOffsetX: 0, shadowOffsetY: 0, filter: 'none', filterChain: [],
      clips: [],
    };
  }
  function cloneState(s) {
    const c = Object.assign({}, s);
    c.m = s.m.slice(); c.lineDash = s.lineDash.slice(); // clips 陣列不可變（clip() 會 concat 新陣列），可共用
    return c;
  }
  const TEXT_PROPS = ['textAlign', 'textBaseline', 'direction', 'letterSpacing', 'wordSpacing', 'fontKerning', 'fontStretch', 'fontVariantCaps', 'textRendering'];

  function make2D(w, h) {
    if (typeof document !== 'undefined') { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }
    return new OffscreenCanvas(w, h);
  }
  function sourceSize(img) {
    if (typeof HTMLImageElement !== 'undefined' && img instanceof HTMLImageElement) return [img.naturalWidth, img.naturalHeight];
    if (typeof HTMLVideoElement !== 'undefined' && img instanceof HTMLVideoElement) return [img.videoWidth, img.videoHeight];
    if (typeof VideoFrame !== 'undefined' && img instanceof VideoFrame) return [img.displayWidth, img.displayHeight];
    if (typeof SVGImageElement !== 'undefined' && img instanceof SVGImageElement) return [img.width.baseVal.value, img.height.baseVal.value];
    return [img.width, img.height];
  }
  /**
   * 把影像來源正規化：context 物件 → 它的 canvas 元素。
   *   WebGL2D context 以 CSS 尺寸當作影像大小（scale = pixelRatio），紋理保留實體像素解析度。
   */
  function resolveSource(src) {
    if (src instanceof WebGL2DRenderingContext) return { el: src.canvas, scale: src._pr };
    const isCtx = (name) => typeof globalThis[name] === 'function' && src instanceof globalThis[name];
    if (src && ['CanvasRenderingContext2D', 'OffscreenCanvasRenderingContext2D', 'WebGLRenderingContext',
      'WebGL2RenderingContext', 'ImageBitmapRenderingContext'].some(isCtx)) return { el: src.canvas, scale: 1 };
    return { el: src, scale: 1 };
  }
  function isStatic(img) {
    return (typeof HTMLImageElement !== 'undefined' && img instanceof HTMLImageElement) ||
      (typeof ImageBitmap !== 'undefined' && img instanceof ImageBitmap);
  }

  /* =====================================================================
   * WebGL2DRenderingContext
   * ===================================================================== */
  class WebGL2DRenderingContext {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      const attrs = {
        alpha: opts.alpha !== false, antialias: opts.antialias !== false, stencil: true, depth: false,
        premultipliedAlpha: true, preserveDrawingBuffer: opts.preserveDrawingBuffer !== false,
        desynchronized: !!opts.desynchronized, powerPreference: opts.powerPreference || 'default',
      };
      let gl = null;
      if (opts.webgl2 !== false) gl = canvas.getContext('webgl2', attrs);
      this.isWebGL2 = !!gl;
      if (!gl) gl = canvas.getContext('webgl', attrs) || canvas.getContext('experimental-webgl', attrs);
      if (!gl) throw new Error('WebGL2D: 此瀏覽器或 canvas 無法建立 WebGL context');
      this.gl = gl;
      this._attrs = attrs;
      const pr = opts.pixelRatio === 'auto' ? (typeof devicePixelRatio === 'number' ? devicePixelRatio : 1) : +opts.pixelRatio;
      this._pr = pr > 0 && Number.isFinite(pr) ? pr : 1;
      if (!gl.getContextAttributes().stencil) console.warn('WebGL2D: 沒有 stencil buffer，fill/clip 將無法正確運作');

      // program
      const sh = (type, src) => {
        const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('WebGL2D shader: ' + gl.getShaderInfoLog(s));
        return s;
      };
      const p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
      gl.bindAttribLocation(p, 0, 'a_pos'); gl.bindAttribLocation(p, 1, 'a_uv');
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('WebGL2D link: ' + gl.getProgramInfoLog(p));
      gl.useProgram(p);
      this._prog = p;
      this._u = {};
      for (const n of ['u_res', 'u_paintInv', 'u_type', 'u_color', 'u_g0', 'u_g1', 'u_ramp', 'u_tex', 'u_mask', 'u_useMask', 'u_alpha', 'u_patSize', 'u_repeat', 'u_patFlip', 'u_quality', 'u_texSize', 'u_dx', 'u_dy'])
        this._u[n] = gl.getUniformLocation(p, n);
      gl.uniform1i(this._u.u_ramp, 0); gl.uniform1i(this._u.u_tex, 1); gl.uniform1i(this._u.u_mask, 2);

      this._vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this._vbo);
      gl.enableVertexAttribArray(0); gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);

      this._dummy = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this._dummy);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
      for (let i = 0; i < 3; i++) { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, this._dummy); }
      gl.disable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE); gl.disable(gl.SCISSOR_TEST);

      this._imgCache = new WeakMap();
      this._dynTex = null;      // canvas / video 來源共用的暫存紋理
      this._textCache = new Map();
      this._mcanvas = make2D(1, 1);
      this._mctx = this._mcanvas.getContext('2d');
      this._tcanvas = make2D(1, 1);
      this._tctx = this._tcanvas.getContext('2d', { willReadFrequently: false });

      this._state = defaultState();
      this._stack = [];
      this._path = new FlatPath(this._pr);
      this._w = -1; this._h = -1;
      this._prepare();
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    }

    /* ---------------- 內部：尺寸 / 模板 ---------------- */
    _prepare() {
      const gl = this.gl, w = this.canvas.width, h = this.canvas.height, pr = this._pr;
      if (w !== this._w || h !== this._h || pr !== this._prUsed) {
        this._w = w; this._h = h; this._prUsed = pr;
        gl.viewport(0, 0, w, h);                       // 實體像素
        const lw = w / pr, lh = h / pr;                // 邏輯像素（繪圖座標）
        gl.uniform2f(this._u.u_res, lw, lh);
        this._full = new Float32Array([0, 0, 0, 0, lw, 0, 0, 0, lw, lh, 0, 0, 0, 0, 0, 0, lw, lh, 0, 0, 0, lh, 0, 0]);
        this._rebuildClip();
      }
    }
    _rebuildClip() {
      const gl = this.gl;
      gl.stencilMask(0xff); gl.clearStencil(0x80); gl.clear(gl.STENCIL_BUFFER_BIT);
      for (const c of this._state.clips) this._applyClip(c);
      this._clipKey = this._state.clips;
    }
    _applyClip(c) {
      const gl = this.gl, fan = fanGeometry(c.subs);
      gl.enable(gl.STENCIL_TEST);
      gl.colorMask(false, false, false, false);
      this._solid(0, 0, 0, 0);
      gl.stencilFunc(gl.ALWAYS, 0, 0xff); gl.stencilMask(0x7f);
      if (c.rule === 'evenodd') gl.stencilOp(gl.KEEP, gl.KEEP, gl.INVERT);
      else { gl.stencilOpSeparate(gl.FRONT, gl.KEEP, gl.KEEP, gl.INCR_WRAP); gl.stencilOpSeparate(gl.BACK, gl.KEEP, gl.KEEP, gl.DECR_WRAP); }
      if (fan.count) this._drawVerts(new Float32Array(fan.v));
      // 環繞數為 0 的地方清除裁切位元
      gl.stencilFunc(gl.EQUAL, 0, c.rule === 'evenodd' ? 0x01 : 0x7f);
      gl.stencilMask(0x80); gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
      this._drawVerts(this._full);
      // 清除低位元
      gl.stencilFunc(gl.ALWAYS, 0, 0xff); gl.stencilMask(0x7f); gl.stencilOp(gl.ZERO, gl.ZERO, gl.ZERO);
      this._drawVerts(this._full);
      gl.colorMask(true, true, true, true);
    }
    _drawVerts(f32) {
      const gl = this.gl, b = this._fxBox;
      if (b) {
        for (let i = 0; i < f32.length; i += 4) {
          const x = f32[i], y = f32[i + 1];
          if (x < b[0]) b[0] = x; if (y < b[1]) b[1] = y; if (x > b[2]) b[2] = x; if (y > b[3]) b[3] = y;
        }
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this._vbo);
      gl.bufferData(gl.ARRAY_BUFFER, f32, gl.STREAM_DRAW);
      gl.drawArrays(gl.TRIANGLES, 0, f32.length / 4);
    }

    /* ---------------- 內部：混合 / 上色 ---------------- */
    _blend() {
      const gl = this.gl, op = this._state.globalCompositeOperation;
      gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD);
      const O = gl.ONE, Z = gl.ZERO, SA = gl.SRC_ALPHA, ISA = gl.ONE_MINUS_SRC_ALPHA, DA = gl.DST_ALPHA, IDA = gl.ONE_MINUS_DST_ALPHA;
      switch (op) {
        case 'source-in': gl.blendFunc(DA, Z); break;
        case 'source-out': gl.blendFunc(IDA, Z); break;
        case 'source-atop': gl.blendFunc(DA, ISA); break;
        case 'destination-over': gl.blendFunc(IDA, O); break;
        case 'destination-in': gl.blendFunc(Z, SA); break;
        case 'destination-out': gl.blendFunc(Z, ISA); break;
        case 'destination-atop': gl.blendFunc(IDA, SA); break;
        case 'lighter': gl.blendFunc(O, O); break;
        case 'copy': gl.blendFunc(O, Z); break;
        case 'xor': gl.blendFunc(IDA, ISA); break;
        case 'screen': gl.blendFunc(O, gl.ONE_MINUS_SRC_COLOR); break;
        default: gl.blendFunc(O, ISA);   // source-over；混合模式在離屏圖層內也以 source-over 繪製
      }
    }
    _solid(r, g, b, a) {
      const gl = this.gl, u = this._u;
      gl.uniform1i(u.u_type, 0); gl.uniform4f(u.u_color, r, g, b, a);
      gl.uniform1f(u.u_alpha, 1); gl.uniform1i(u.u_useMask, 0);
    }
    _filter(tex) {
      const gl = this.gl, f = this._state.imageSmoothingEnabled ? gl.LINEAR : gl.NEAREST;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
    }
    /** 設定 paint uniforms。回傳 true 代表「什麼都不畫」（例如退化的漸層） */
    _paint(style, opt = {}) {
      const gl = this.gl, u = this._u, s = this._state;
      gl.uniform1f(u.u_alpha, s.globalAlpha);
      gl.uniform1i(u.u_useMask, opt.mask ? 1 : 0);
      gl.uniform1f(u.u_patFlip, 0);
      if (opt.mask) { gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, opt.mask); }
      if (opt.image) {
        gl.uniform1i(u.u_type, 5);
        // opt.dxy：每個實體像素對應的 uv 變化量（drawImage 提供）；沒有時視為 1:1 的雙線性複製
        if (opt.dxy) this._quality(opt.size, opt.dxy); else gl.uniform1i(u.u_quality, 1);
        gl.activeTexture(gl.TEXTURE1); this._filter(opt.image);
        gl.activeTexture(gl.TEXTURE0);
        return false;
      }
      const im = inv(s.m) || [1, 0, 0, 1, 0, 0];
      const setInv = (m) => gl.uniformMatrix3fv(u.u_paintInv, false, [m[0], m[1], 0, m[2], m[3], 0, m[4], m[5], 1]);
      let empty = false;
      if (style instanceof CanvasGradient) {
        const g = style, P = g._p;
        setInv(im);
        if (!g._stops.length) empty = true;
        else if (g._type === 'linear') {
          if (P[0] === P[2] && P[1] === P[3]) empty = true;
          gl.uniform1i(u.u_type, 1); gl.uniform4f(u.u_g0, P[0], P[1], P[2], P[3]);
        } else if (g._type === 'radial') {
          if (P[0] === P[3] && P[1] === P[4] && P[2] === P[5]) empty = true;
          gl.uniform1i(u.u_type, 2); gl.uniform4f(u.u_g0, P[0], P[1], P[2], 0); gl.uniform4f(u.u_g1, P[3], P[4], P[5], 0);
        } else {
          gl.uniform1i(u.u_type, 3); gl.uniform4f(u.u_g0, P[1], P[2], P[0], 0);
        }
        if (!empty) { gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, g._texture(gl)); }
      } else if (style instanceof CanvasPattern) {
        const pi = inv(style._m);
        if (!pi) empty = true;
        const P = pi ? mul(pi, im) : im;
        setInv(P);
        const kx = 1 / (style._w * this._pr), ky = 1 / (style._h * this._pr);
        this._quality([style._tw, style._th], [P[0] * kx, P[1] * ky, P[2] * kx, P[3] * ky]);
        gl.uniform1i(u.u_type, 4);
        gl.uniform2f(u.u_patSize, style._w, style._h);
        const r = style._rep;
        gl.uniform2f(u.u_repeat, r === 'repeat' || r === 'repeat-x' ? 1 : 0, r === 'repeat' || r === 'repeat-y' ? 1 : 0);
        gl.uniform1f(u.u_patFlip, style._flipY ? 1 : 0);
        gl.activeTexture(gl.TEXTURE1); this._filter(style._tex);
      } else {
        const c = style.color, a = c[3];
        if (a === 0 && !UNBOUNDED.has(s.globalCompositeOperation)) empty = true;
        gl.uniform1i(u.u_type, 0); gl.uniform4f(u.u_color, (c[0] / 255) * a, (c[1] / 255) * a, (c[2] / 255) * a, a);
      }
      gl.activeTexture(gl.TEXTURE0);
      if (empty) { gl.uniform1i(u.u_type, 0); gl.uniform4f(u.u_color, 0, 0, 0, 0); }
      return empty;
    }
    /** 依 imageSmoothingEnabled / imageSmoothingQuality 設定影像取樣方式 */
    _quality(size, dxy) {
      const gl = this.gl, u = this._u, s = this._state;
      const q = !s.imageSmoothingEnabled ? 0 : s.imageSmoothingQuality === 'high' ? 3 : s.imageSmoothingQuality === 'medium' ? 2 : 1;
      gl.uniform1i(u.u_quality, q);
      gl.uniform2f(u.u_texSize, size[0], size[1]);
      gl.uniform2f(u.u_dx, dxy[0], dxy[1]);
      gl.uniform2f(u.u_dy, dxy[2], dxy[3]);
    }
    _unbounded() { return UNBOUNDED.has(this._state.globalCompositeOperation); }
    _syncClip() { if (this._clipKey !== this._state.clips) this._rebuildClip(); }

    /** 凸多邊形（矩形、影像、文字方塊）：只需裁切測試，不用寫入模板 */
    _drawConvex(verts, style, opt) {
      if (this._unbounded()) return this._drawStenciled(verts, 'replace', verts, style, opt);
      const gl = this.gl;
      gl.enable(gl.STENCIL_TEST); gl.stencilMask(0); gl.stencilFunc(gl.EQUAL, 0x80, 0x80); gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
      this._blend();
      if (this._paint(style, opt)) return;
      this._drawVerts(verts);
    }
    /** Stencil-then-Cover */
    _drawStenciled(stencilVerts, mode, coverVerts, style, opt) {
      const gl = this.gl, unb = this._unbounded();
      gl.enable(gl.STENCIL_TEST);
      gl.colorMask(false, false, false, false);
      this._solid(0, 0, 0, 0);
      gl.stencilMask(0x7f); gl.stencilFunc(gl.ALWAYS, 1, 0xff);
      if (mode === 'nonzero') { gl.stencilOpSeparate(gl.FRONT, gl.KEEP, gl.KEEP, gl.INCR_WRAP); gl.stencilOpSeparate(gl.BACK, gl.KEEP, gl.KEEP, gl.DECR_WRAP); }
      else if (mode === 'evenodd') gl.stencilOp(gl.KEEP, gl.KEEP, gl.INVERT);
      else gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
      this._drawVerts(stencilVerts);

      gl.colorMask(true, true, true, true);
      if (mode === 'nonzero') gl.stencilFunc(gl.LESS, 0x80, 0xff); // stencil > 0x80：裁切位元已設且環繞數 ≠ 0
      else gl.stencilFunc(gl.EQUAL, 0x81, 0x81);
      const k = unb ? gl.KEEP : gl.ZERO;
      gl.stencilOp(k, k, k);
      this._blend();
      const empty = this._paint(style, opt);
      if (empty && !unb) gl.colorMask(false, false, false, false);
      this._drawVerts(coverVerts);
      gl.colorMask(true, true, true, true);

      if (unb) {
        // 形狀以外（裁切區內）視為透明來源 → 清除
        gl.stencilFunc(gl.EQUAL, 0x80, 0x80 | (mode === 'nonzero' ? 0x7f : 0x01));
        gl.stencilMask(0); gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
        gl.blendFunc(gl.ONE, gl.ZERO); gl.blendEquation(gl.FUNC_ADD);
        this._solid(0, 0, 0, 0);
        this._drawVerts(this._full);
        gl.colorMask(false, false, false, false);
        gl.stencilFunc(gl.ALWAYS, 0, 0xff); gl.stencilMask(0x7f); gl.stencilOp(gl.ZERO, gl.ZERO, gl.ZERO);
        this._drawVerts(this._full);
        gl.colorMask(true, true, true, true);
      }
    }
    _quad(x, y, w, h, u0 = 0, v0 = 0, u1 = 1, v1 = 1) {
      const m = this._state.m;
      const T = (px, py) => [m[0] * px + m[2] * py + m[4], m[1] * px + m[3] * py + m[5]];
      const a = T(x, y), b = T(x + w, y), c = T(x + w, y + h), d = T(x, y + h);
      return new Float32Array([a[0], a[1], u0, v0, b[0], b[1], u1, v0, c[0], c[1], u1, v1, a[0], a[1], u0, v0, c[0], c[1], u1, v1, d[0], d[1], u0, v1]);
    }
    _fillSubs(subs, rule, style) {
      const fan = fanGeometry(subs);
      if (!fan.count) { if (this._unbounded()) this._drawStenciled(new Float32Array(0), 'evenodd', new Float32Array(0), style); return; }
      this._drawStenciled(new Float32Array(fan.v), rule === 'evenodd' ? 'evenodd' : 'nonzero', fan.cover(), style);
    }
    _strokeSubs(subs) {
      const g = buildStroke(subs, this._state.m, this._state, this._pr);
      if (!g || !g.count) return;
      this._drawStenciled(new Float32Array(g.v), 'replace', g.cover(), this._state.stroke);
    }
    _pathArgs(a) {
      // (path?, ...rest)
      if (a[0] instanceof Path2D) return [a[0], a.slice(1)];
      if (a[0] && typeof a[0] === 'object' && !(a[0] instanceof Path2D)) throw new TypeError('WebGL2D 只接受 WebGL2D.Path2D 物件');
      return [null, a];
    }
    _subsFor(path) {
      if (!path) return this._path.subs;
      const f = new FlatPath(this._pr); replay(f, path._ops, this._state.m); return f.subs;
    }

    /* =================== 狀態 =================== */
    save() { this._stack.push(cloneState(this._state)); }
    restore() {
      if (!this._stack.length) return;
      this._state = this._stack.pop();
      this._syncClip();
    }
    reset() {
      this._state = defaultState(); this._stack = []; this._path = new FlatPath(this._pr);
      const gl = this.gl; gl.colorMask(true, true, true, true); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      this._w = -1; this._prepare();
    }
    /** 每個繪圖座標單位對應幾個實體像素。改變後座標系統立即生效，不會重新取樣既有內容 */
    get pixelRatio() { return this._pr; }
    set pixelRatio(v) { v = +v; if (v > 0 && Number.isFinite(v)) this._pr = v; }
    isContextLost() { return this.gl.isContextLost(); }
    getContextAttributes() {
      return { alpha: this._attrs.alpha, colorSpace: 'srgb', desynchronized: this._attrs.desynchronized, willReadFrequently: false };
    }

    /* =================== 變換 =================== */
    scale(x, y) { if (fin(x, y)) this._state.m = mul(this._state.m, [x, 0, 0, y, 0, 0]); }
    rotate(a) { if (fin(a)) { const c = Math.cos(a), s = Math.sin(a); this._state.m = mul(this._state.m, [c, s, -s, c, 0, 0]); } }
    translate(x, y) { if (fin(x, y)) this._state.m = mul(this._state.m, [1, 0, 0, 1, x, y]); }
    transform(a, b, c, d, e, f) { if (fin(a, b, c, d, e, f)) this._state.m = mul(this._state.m, [a, b, c, d, e, f]); }
    setTransform(a, b, c, d, e, f) {
      if (arguments.length <= 1) { this._state.m = toMatrix(a); return; }
      if (fin(a, b, c, d, e, f)) this._state.m = [a, b, c, d, e, f];
    }
    resetTransform() { this._state.m = [1, 0, 0, 1, 0, 0]; }
    getTransform() { return makeDOMMatrix(this._state.m.slice()); }

    /* =================== 合成 / 樣式 =================== */
    get globalAlpha() { return this._state.globalAlpha; }
    set globalAlpha(v) { v = +v; if (v >= 0 && v <= 1) this._state.globalAlpha = v; }
    get globalCompositeOperation() { return this._state.globalCompositeOperation; }
    set globalCompositeOperation(v) {
      if (COMPOSITE.has(v)) this._state.globalCompositeOperation = v;
    }
    _setStyle(key, v) {
      if (v instanceof CanvasGradient || v instanceof CanvasPattern) { this._state[key] = v; return; }
      if (typeof v !== 'string') return;
      const c = parseColor(v);
      if (c) this._state[key] = { color: c };
    }
    _getStyle(key) { const s = this._state[key]; return s.color ? serializeColor(s.color) : s; }
    get fillStyle() { return this._getStyle('fill'); }
    set fillStyle(v) { this._setStyle('fill', v); }
    get strokeStyle() { return this._getStyle('stroke'); }
    set strokeStyle(v) { this._setStyle('stroke', v); }

    createLinearGradient(x0, y0, x1, y1) {
      if (!fin(x0, y0, x1, y1)) throw new TypeError('createLinearGradient: non-finite argument');
      return new CanvasGradient(this, 'linear', [x0, y0, x1, y1]);
    }
    createRadialGradient(x0, y0, r0, x1, y1, r1) {
      if (!fin(x0, y0, r0, x1, y1, r1)) throw new TypeError('createRadialGradient: non-finite argument');
      if (r0 < 0 || r1 < 0) throw domError('The radius provided is negative.', 'IndexSizeError');
      return new CanvasGradient(this, 'radial', [x0, y0, r0, x1, y1, r1]);
    }
    createConicGradient(startAngle, x, y) {
      if (!fin(startAngle, x, y)) throw new TypeError('createConicGradient: non-finite argument');
      return new CanvasGradient(this, 'conic', [startAngle, x, y]);
    }
    createPattern(image, repetition) {
      const rep = repetition === null || repetition === undefined || repetition === '' ? 'repeat' : String(repetition);
      if (!['repeat', 'repeat-x', 'repeat-y', 'no-repeat'].includes(rep)) throw domError(`Invalid repetition '${repetition}'`, 'SyntaxError');
      const { el, scale } = resolveSource(image);
      if (!el) return null;
      const [tw, th] = sourceSize(el);
      if (!tw || !th) return null;
      if (typeof HTMLImageElement !== 'undefined' && el instanceof HTMLImageElement && !el.complete) return null;
      if (typeof HTMLVideoElement !== 'undefined' && el instanceof HTMLVideoElement && el.readyState < 2) return null;
      const gl = this.gl, tex = gl.createTexture();
      let flip = false;
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      if (el === this.canvas) {
        if (this._curFB) gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.copyTexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 0, 0, tw, th, 0); flip = true;
        if (this._curFB) gl.bindFramebuffer(gl.FRAMEBUFFER, this._curFB);
      } else { gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, el); }
      this._texParams();
      gl.activeTexture(gl.TEXTURE0);
      return new CanvasPattern(tex, tw / scale, th / scale, rep, flip, tw, th);
    }
    _texParams() {
      const gl = this.gl;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    }

    /* =================== 線條樣式 =================== */
    get lineWidth() { return this._state.lineWidth; }
    set lineWidth(v) { v = +v; if (v > 0 && Number.isFinite(v)) this._state.lineWidth = v; }
    get lineCap() { return this._state.lineCap; }
    set lineCap(v) { if (['butt', 'round', 'square'].includes(v)) this._state.lineCap = v; }
    get lineJoin() { return this._state.lineJoin; }
    set lineJoin(v) { if (['miter', 'round', 'bevel'].includes(v)) this._state.lineJoin = v; }
    get miterLimit() { return this._state.miterLimit; }
    set miterLimit(v) { v = +v; if (v > 0 && Number.isFinite(v)) this._state.miterLimit = v; }
    get lineDashOffset() { return this._state.lineDashOffset; }
    set lineDashOffset(v) { v = +v; if (Number.isFinite(v)) this._state.lineDashOffset = v; }
    setLineDash(segments) {
      const a = Array.from(segments, Number);
      if (a.some((v) => !Number.isFinite(v) || v < 0)) return;
      this._state.lineDash = a.length % 2 ? a.concat(a) : a;
    }
    getLineDash() { return this._state.lineDash.slice(); }

    /* =================== 影像平滑 / 陰影 / 濾鏡 =================== */
    get imageSmoothingEnabled() { return this._state.imageSmoothingEnabled; }
    set imageSmoothingEnabled(v) { this._state.imageSmoothingEnabled = !!v; }
    get imageSmoothingQuality() { return this._state.imageSmoothingQuality; }
    set imageSmoothingQuality(v) { if (['low', 'medium', 'high'].includes(v)) this._state.imageSmoothingQuality = v; }
    get shadowBlur() { return this._state.shadowBlur; }
    set shadowBlur(v) { v = +v; if (v >= 0 && Number.isFinite(v)) this._state.shadowBlur = v; }
    get shadowColor() { return this._state.shadowColor; }
    set shadowColor(v) { const c = parseColor(String(v)); if (c) this._state.shadowColor = serializeColor(c); }
    get shadowOffsetX() { return this._state.shadowOffsetX; }
    set shadowOffsetX(v) { v = +v; if (Number.isFinite(v)) this._state.shadowOffsetX = v; }
    get shadowOffsetY() { return this._state.shadowOffsetY; }
    set shadowOffsetY(v) { v = +v; if (Number.isFinite(v)) this._state.shadowOffsetY = v; }
    /** CSS filter 字串、ShaderFilter，或兩者混合的陣列（依序套用）。無效的值會被忽略 */
    get filter() { return this._state.filter; }
    set filter(v) {
      const chain = buildFilterChain(v);
      if (!chain) return;
      this._state.filter = Array.isArray(v) ? v.slice() : v;
      this._state.filterChain = chain;
    }
    /** 編譯自訂濾鏡。source 需定義 `vec4 effect(vec2 uv)`，回傳預乘 alpha 的顏色 */
    createShaderFilter(source, uniforms) { return new ShaderFilter(this, source, uniforms); }
    /** 對畫布上已有的內容套用濾鏡（不受變換影響；受裁切影響）。矩形以 CSS 像素計，預設為整個畫布 */
    applyFilter(filter, x, y, w, h) {
      const chain = buildFilterChain(filter);
      if (!chain) throw new TypeError('applyFilter：無效的濾鏡');
      this._prepare(); this._syncClip();
      const gl = this.gl, fx = this._fxEnsure(), pr = this._pr, lw = this._w / pr, lh = this._h / pr;
      if (x === undefined) { x = 0; y = 0; w = lw; h = lh; }
      if (!fin(x, y, w, h) || !w || !h) return;
      if (w < 0) { x += w; w = -w; } if (h < 0) { y += h; h = -h; }
      this._unbindUnits();
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, fx.tex[0]);
      gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, fx.w, fx.h);
      const res = this._runChain(chain, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.enable(gl.STENCIL_TEST); gl.stencilMask(0); gl.stencilFunc(gl.EQUAL, 0x80, 0x80); gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
      gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD); gl.blendFunc(gl.ONE, gl.ZERO);
      const u = this._u;
      gl.uniform1i(u.u_type, 5); gl.uniform1f(u.u_alpha, 1); gl.uniform1i(u.u_useMask, 0); gl.uniform1i(u.u_quality, 1);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, fx.tex[res]); gl.activeTexture(gl.TEXTURE0);
      const x1 = x + w, y1 = y + h, U = (px) => px / lw, V = (py) => 1 - py / lh;
      this._drawVerts(new Float32Array([x, y, U(x), V(y), x1, y, U(x1), V(y), x1, y1, U(x1), V(y1), x, y, U(x), V(y), x1, y1, U(x1), V(y1), x, y1, U(x), V(y1)]));
      this._unbindUnits();
    }
    _effects() {
      const s = this._state;
      let chain = s.filterChain;
      if (s.shadowBlur > 0 || s.shadowOffsetX || s.shadowOffsetY) {
        const c = parseColor(s.shadowColor);
        if (c && c[3] > 0) chain = chain.concat([{ t: 'shadow', ox: s.shadowOffsetX, oy: s.shadowOffsetY, sigma: s.shadowBlur / 2, color: c }]);
      }
      return chain.length ? chain : null;
    }
    _unbindUnits() {
      const gl = this.gl;
      for (let i = 0; i < 8; i++) { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, this._dummy); }
      gl.activeTexture(gl.TEXTURE0);
    }
    _fxEnsure() {
      const gl = this.gl, w = this._w, h = this._h;
      if (!this._fxp) {
        this._fxp = {};
        for (const k in FX_FS) this._fxp[k] = compileProgram(gl, FX_VS, FX_FS[k], 'filter:' + k);
      }
      if (this._fx && this._fx.w === w && this._fx.h === h) return this._fx;
      const old = this._fx;
      if (old) {
        old.tex.forEach((t) => gl.deleteTexture(t)); old.fb.forEach((f) => gl.deleteFramebuffer(f));
        old.rbs.forEach((r) => gl.deleteRenderbuffer(r)); if (old.layer !== old.fb[0]) gl.deleteFramebuffer(old.layer);
      }
      const fx = { w, h, tex: [], fb: [], rbs: [], msaa: false, layer: null };
      for (let i = 0; i < 3; i++) {
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        this._texParams();
        const f = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, f);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
        fx.tex.push(t); fx.fb.push(f);
      }
      const samples = this.isWebGL2 && this._attrs.antialias ? Math.min(4, gl.getParameter(gl.MAX_SAMPLES)) : 0;
      if (samples > 0) {
        // WebGL 2：多重取樣圖層（保留反鋸齒），繪製後 blit 到 tex[0]
        const color = gl.createRenderbuffer(), ds = gl.createRenderbuffer(), f = gl.createFramebuffer();
        gl.bindRenderbuffer(gl.RENDERBUFFER, color);
        gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.RGBA8, w, h);
        gl.bindRenderbuffer(gl.RENDERBUFFER, ds);
        gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.DEPTH24_STENCIL8, w, h);
        gl.bindFramebuffer(gl.FRAMEBUFFER, f);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, color);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, ds);
        fx.layer = f; fx.msaa = true; fx.rbs.push(color, ds);
      } else {
        const ds = gl.createRenderbuffer();
        gl.bindRenderbuffer(gl.RENDERBUFFER, ds);
        gl.renderbufferStorage(gl.RENDERBUFFER, this.isWebGL2 ? gl.DEPTH24_STENCIL8 : gl.DEPTH_STENCIL, w, h);
        gl.bindFramebuffer(gl.FRAMEBUFFER, fx.fb[0]);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, ds);
        fx.layer = fx.fb[0]; fx.rbs.push(ds);
      }
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) console.warn('WebGL2D: 濾鏡圖層 framebuffer 不完整');
      gl.bindRenderbuffer(gl.RENDERBUFFER, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this._curFB || null);
      this._fx = fx;
      return fx;
    }
    /** 把 fn 的繪圖結果畫進圖層，套用 chain 後依裁切 / 透明度 / 合成模式貼回畫面 */
    _filtered(chain, fn) {
      const gl = this.gl, s = this._state;
      this._prepare(); this._syncClip();
      const fx = this._fxEnsure();
      this._unbindUnits();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fx.layer); this._curFB = fx.layer;
      gl.colorMask(true, true, true, true); gl.stencilMask(0xff);
      gl.clearColor(0, 0, 0, 0); gl.clearStencil(0x80); gl.clear(gl.COLOR_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
      const op = s.globalCompositeOperation, ga = s.globalAlpha;
      // 自訂 shader 可能讀取任意位置、無界合成需要整個畫布 → 不限制範圍
      const full = UNBOUNDED.has(op) || chain.some((c) => c.t === 'custom');
      this._fxBox = full ? null : [Infinity, Infinity, -Infinity, -Infinity];
      s.globalCompositeOperation = 'source-over'; s.globalAlpha = 1;
      let box;
      try { fn(); } finally {
        s.globalCompositeOperation = op; s.globalAlpha = ga;
        this._curFB = null;
        box = this._fxBox; this._fxBox = null;
      }
      const pr = this._pr, W = fx.w, H = fx.h;
      let sx = 0, sy = 0, sw = W, sh = H;
      if (box) {
        if (!(box[2] > box[0])) { gl.bindFramebuffer(gl.FRAMEBUFFER, null); return; }   // 沒有畫任何東西
        let m = 2;   // 濾鏡會擴散的範圍（CSS 像素）
        for (const c of chain) {
          if (c.t === 'blur') m += 3 * c.sigma + 2;
          else if (c.t === 'shadow') m += 3 * c.sigma + Math.max(Math.abs(c.ox), Math.abs(c.oy)) + 2;
        }
        const x0 = Math.max(0, Math.floor((box[0] - m) * pr)), y0 = Math.max(0, Math.floor((box[1] - m) * pr));
        const x1 = Math.min(W, Math.ceil((box[2] + m) * pr)), y1 = Math.min(H, Math.ceil((box[3] + m) * pr));
        if (x1 <= x0 || y1 <= y0) { gl.bindFramebuffer(gl.FRAMEBUFFER, null); return; }
        sx = x0; sw = x1 - x0; sy = H - y1; sh = y1 - y0;   // GL 座標（y 向上）
        gl.enable(gl.SCISSOR_TEST); gl.scissor(sx, sy, sw, sh);
      }
      const boxUV = [sx / W, sy / H, (sx + sw) / W, (sy + sh) / H];
      try {
      if (fx.msaa) {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fx.layer); gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, fx.fb[0]);
        gl.blitFramebuffer(0, 0, fx.w, fx.h, 0, 0, fx.w, fx.h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null); gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
      }
      this._unbindUnits();
      const res = this._runChain(chain, 0, boxUV);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      const lw = W / pr, lh = H / pr;
      const q = new Float32Array([0, 0, 0, 1, lw, 0, 1, 1, lw, lh, 1, 0, 0, 0, 0, 1, lw, lh, 1, 0, 0, lh, 0, 0]);
      const mode = BLEND_MODES[op];
      if (mode) {
        // 混合模式：複製目的地區域 → 混合 shader 計算結果（僅寫入裁切區內）
        const D = [0, 1, 2].find((i) => i !== res), P = this._fxp.blend;
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, fx.tex[D]);
        gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, sx, sy, sx, sy, sw, sh);
        gl.useProgram(P.p);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, fx.tex[res]);
        gl.uniform1i(P.u('u_src'), 0); gl.uniform1i(P.u('u_dst'), 1);
        gl.uniform1i(P.u('u_mode'), mode); gl.uniform1f(P.u('u_alpha'), ga);
        gl.disable(gl.BLEND); gl.colorMask(true, true, true, true);
        gl.enable(gl.STENCIL_TEST); gl.stencilMask(0); gl.stencilFunc(gl.EQUAL, 0x80, 0x80); gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
        try { this._drawVerts(FX_QUAD); } finally { gl.useProgram(this._prog); gl.enable(gl.BLEND); }
      } else {
        const saveM = s.m; s.m = [1, 0, 0, 1, 0, 0];
        try { this._drawConvex(q, null, { image: fx.tex[res] }); } finally { s.m = saveM; }
      }
      } finally { gl.disable(gl.SCISSOR_TEST); }
      this._unbindUnits();
    }
    _fxPass(prog, src, dst, setup) {
      const gl = this.gl, fx = this._fx;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fx.fb[dst]);
      gl.useProgram(prog.p);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, fx.tex[src]);
      gl.uniform1i(prog.u('u_src'), 0);
      if (setup) setup(prog);
      this._drawVerts(FX_QUAD);
    }
    _runChain(chain, cur, boxUV = [0, 0, 1, 1]) {
      const gl = this.gl, fx = this._fx, P = this._fxp, pr = this._pr, W = fx.w, H = fx.h;
      for (const k of ['blur', 'tint']) { gl.useProgram(P[k].p); gl.uniform4fv(P[k].u('u_box'), boxUV); }
      gl.disable(gl.STENCIL_TEST); gl.disable(gl.BLEND); gl.colorMask(true, true, true, true);
      const other = (...ex) => [0, 1, 2].find((i) => !ex.includes(i));
      const blur = (src, tmp, dst, sigma) => {
        this._fxPass(P.blur, src, tmp, (p) => { gl.uniform2f(p.u('u_dir'), 1 / W, 0); gl.uniform1f(p.u('u_sigma'), sigma); });
        this._fxPass(P.blur, tmp, dst, (p) => { gl.uniform2f(p.u('u_dir'), 0, 1 / H); gl.uniform1f(p.u('u_sigma'), sigma); });
      };
      try {
        for (const op of chain) {
          if (op.t === 'blur') {
            const sg = op.sigma * pr;
            if (sg < 0.05) continue;
            const a = other(cur), b = other(cur, a);
            blur(cur, a, b, sg); cur = b;
          } else if (op.t === 'matrix') {
            const a = other(cur);
            this._fxPass(P.matrix, cur, a, (p) => { gl.uniformMatrix4fv(p.u('u_m'), false, op.m); gl.uniform4fv(p.u('u_off'), op.off); });
            cur = a;
          } else if (op.t === 'shadow') {
            const a = other(cur), b = other(cur, a), c = op.color;
            this._fxPass(P.tint, cur, a, (p) => {
              gl.uniform4f(p.u('u_color'), (c[0] / 255) * c[3], (c[1] / 255) * c[3], (c[2] / 255) * c[3], c[3]);
              gl.uniform2f(p.u('u_off'), (op.ox * pr) / W, (-op.oy * pr) / H);
            });
            if (op.sigma * pr >= 0.05) blur(a, b, a, op.sigma * pr);
            this._fxPass(P.over, cur, b, (p) => {
              gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, fx.tex[a]);
              gl.uniform1i(p.u('u_under'), 1); gl.activeTexture(gl.TEXTURE0);
            });
            cur = b;
          } else if (op.t === 'custom') {
            const a = other(cur), f = op.f, p = f._prog;
            this._fxPass(p, cur, a, () => {
              gl.uniform2f(p.u('u_resolution'), W, H);
              gl.uniform2f(p.u('u_size'), W / pr, H / pr);
              gl.uniform1f(p.u('u_pixelRatio'), pr);
              f._apply(gl, 3);
            });
            cur = a;
          }
        }
      } finally {
        gl.useProgram(this._prog);
        gl.enable(gl.BLEND);
      }
      return cur;
    }

    /* =================== 文字樣式 =================== */
    get font() { return this._state.font; }
    set font(v) {
      const c = this._mctx; c.font = this._state.font; c.font = String(v); this._state.font = c.font;
    }
    _syncText(c) {
      c.font = this._state.font;
      for (const k of TEXT_PROPS) if (k in c) c[k] = this._state[k];
    }

    /* =================== 矩形 =================== */
    clearRect(x, y, w, h) {
      if (!fin(x, y, w, h) || !w || !h) return;
      this._prepare(); this._syncClip();
      const gl = this.gl;
      gl.enable(gl.STENCIL_TEST); gl.stencilMask(0); gl.stencilFunc(gl.EQUAL, 0x80, 0x80); gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
      gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD); gl.blendFunc(gl.ZERO, gl.ZERO);
      this._solid(0, 0, 0, 0);
      this._drawVerts(this._quad(x, y, w, h));
    }
    fillRect(x, y, w, h) {
      if (!fin(x, y, w, h)) return;
      this._prepare(); this._syncClip();
      if (!w || !h) { if (this._unbounded()) this._fillSubs([], 'nonzero', this._state.fill); return; }
      const q = this._quad(x, y, w, h);
      this._drawConvex(q, this._state.fill);
    }
    strokeRect(x, y, w, h) {
      if (!fin(x, y, w, h)) return;
      this._prepare(); this._syncClip();
      const f = new FlatPath(this._pr);
      if (!w && !h) return;
      if (!w || !h) { f.moveTo(this._state.m, x, y); f.lineTo(this._state.m, x + w, y + h); }
      else f.rect(this._state.m, x, y, w, h);
      this._strokeSubs(f.subs);
    }

    /* =================== 路徑 =================== */
    beginPath() { this._path = new FlatPath(this._pr); }
    closePath() { this._path.closePath(); }
    moveTo(x, y) { this._path.moveTo(this._state.m, x, y); }
    lineTo(x, y) { this._path.lineTo(this._state.m, x, y); }
    quadraticCurveTo(cpx, cpy, x, y) { this._path.quadraticCurveTo(this._state.m, cpx, cpy, x, y); }
    bezierCurveTo(c1x, c1y, c2x, c2y, x, y) { this._path.bezierCurveTo(this._state.m, c1x, c1y, c2x, c2y, x, y); }
    arc(x, y, r, sa, ea, ccw = false) { this._path.arc(this._state.m, x, y, r, sa, ea, !!ccw); }
    arcTo(x1, y1, x2, y2, r) { this._path.arcTo(this._state.m, x1, y1, x2, y2, r); }
    ellipse(x, y, rx, ry, rot, sa, ea, ccw = false) { this._path.ellipse(this._state.m, x, y, rx, ry, rot, sa, ea, !!ccw); }
    rect(x, y, w, h) { this._path.rect(this._state.m, x, y, w, h); }
    roundRect(x, y, w, h, radii = 0) { this._path.roundRect(this._state.m, x, y, w, h, radii); }

    fill(...a) {
      const [path, rest] = this._pathArgs(a);
      this._prepare(); this._syncClip();
      this._fillSubs(this._subsFor(path), rest[0] === 'evenodd' ? 'evenodd' : 'nonzero', this._state.fill);
    }
    stroke(path) {
      const [p] = this._pathArgs([path]);
      this._prepare(); this._syncClip();
      this._strokeSubs(this._subsFor(p));
    }
    clip(...a) {
      const [path, rest] = this._pathArgs(a);
      this._prepare(); this._syncClip();
      const subs = path ? this._subsFor(path) : this._path.snapshot();
      const c = { subs, rule: rest[0] === 'evenodd' ? 'evenodd' : 'nonzero' };
      this._state.clips = this._state.clips.concat([c]);
      this._applyClip(c);
      this._clipKey = this._state.clips;
    }
    isPointInPath(...a) {
      const [path, rest] = this._pathArgs(a);
      const [x, y, rule] = rest;
      if (!fin(x, y)) return false;
      const w = windingAt(this._subsFor(path), x, y);
      return rule === 'evenodd' ? (w & 1) === 1 : w !== 0;
    }
    isPointInStroke(...a) {
      const [path, rest] = this._pathArgs(a);
      const [x, y] = rest;
      if (!fin(x, y)) return false;
      const g = buildStroke(this._subsFor(path), this._state.m, this._state, this._pr);
      return !!g && pointInTris(g.v, x, y);
    }
    drawFocusIfNeeded() { /* 無焦點環概念，不做事 */ }
    scrollPathIntoView() { /* 不做事 */ }

    /* =================== 影像 =================== */
    /** 取得來源的紋理。回傳 { tex, w, h（影像大小，drawImage 座標單位）, tw, th（texel）, flipY } */
    _source(src) {
      const gl = this.gl;
      const { el: img, scale } = resolveSource(src);
      if (!img) return null;
      const [tw, th] = sourceSize(img);
      if (!tw || !th) return null;
      const w = tw / scale, h = th / scale;
      if (img === this.canvas) {
        if (!this._selfTex) this._selfTex = gl.createTexture();
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this._selfTex);
        if (this._curFB) gl.bindFramebuffer(gl.FRAMEBUFFER, null);   // 繪製濾鏡圖層時，仍從畫面複製
        gl.copyTexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 0, 0, tw, th, 0);
        if (this._curFB) gl.bindFramebuffer(gl.FRAMEBUFFER, this._curFB);
        this._texParams(); gl.activeTexture(gl.TEXTURE0);
        return { tex: this._selfTex, w, h, tw, th, flipY: true };
      }
      if (typeof HTMLImageElement !== 'undefined' && img instanceof HTMLImageElement && !img.complete) return null;
      if (typeof HTMLVideoElement !== 'undefined' && img instanceof HTMLVideoElement && img.readyState < 2) return null;
      let e = this._imgCache.get(img);
      const key = img.currentSrc || img.src || '';
      if (e && isStatic(img) && e.key === key && e.tw === tw && e.th === th) return e;
      // 靜態影像快取紋理；canvas / context / video / VideoFrame 每次重新上傳（內容可能改變）
      const tex = isStatic(img) ? (e ? e.tex : gl.createTexture()) : (this._dynTex || (this._dynTex = gl.createTexture()));
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      this._texParams(); gl.activeTexture(gl.TEXTURE0);
      e = { tex, w, h, tw, th, key, flipY: false };
      if (isStatic(img)) this._imgCache.set(img, e);
      return e;
    }
    drawImage(img, ...a) {
      let sx, sy, sw, sh, dx, dy, dw, dh;
      if (a.length !== 2 && a.length !== 4 && a.length !== 8) throw new TypeError('drawImage: 參數數量必須為 3、5 或 9');
      if (!fin(...a)) return;
      this._prepare(); this._syncClip();
      const src = this._source(img);
      if (!src) return;
      if (a.length === 2) { [dx, dy] = a; sx = 0; sy = 0; sw = dw = src.w; sh = dh = src.h; }
      else if (a.length === 4) { [dx, dy, dw, dh] = a; sx = 0; sy = 0; sw = src.w; sh = src.h; }
      else [sx, sy, sw, sh, dx, dy, dw, dh] = a;
      if (sw < 0) { sx += sw; sw = -sw; } if (sh < 0) { sy += sh; sh = -sh; }
      if (dw < 0) { dx += dw; dw = -dw; } if (dh < 0) { dy += dh; dh = -dh; }
      if (!sw || !sh || !dw || !dh) return;
      const kx = dw / sw, ky = dh / sh;
      if (sx < 0) { dx -= sx * kx; dw += sx * kx; sw += sx; sx = 0; }
      if (sy < 0) { dy -= sy * ky; dh += sy * ky; sh += sy; sy = 0; }
      if (sx + sw > src.w) { dw -= (sx + sw - src.w) * kx; sw = src.w - sx; }
      if (sy + sh > src.h) { dh -= (sy + sh - src.h) * ky; sh = src.h - sy; }
      if (sw <= 0 || sh <= 0) return;
      let v0 = sy / src.h, v1 = (sy + sh) / src.h;
      if (src.flipY) { v0 = 1 - v0; v1 = 1 - v1; }
      const q = this._quad(dx, dy, dw, dh, sx / src.w, v0, (sx + sw) / src.w, v1);
      // 實體像素 → 使用者座標 → uv 的線性部分（用來決定縮放時的取樣範圍）
      const im = inv(this._state.m) || [1, 0, 0, 1, 0, 0], pr = this._pr;
      const au = sw / src.w / dw / pr, av = (v1 - v0) / dh / pr;
      this._drawConvex(q, null, { image: src.tex, size: [src.tw, src.th], dxy: [au * im[0], av * im[1], au * im[2], av * im[3]] });
    }

    /* =================== 像素操作 =================== */
    createImageData(a, b) {
      if (a && typeof a === 'object') return new ImageData(a.width, a.height);
      const w = Math.abs(Math.trunc(a)), h = Math.abs(Math.trunc(b));
      if (!w || !h) throw domError('The source width or height is 0.', 'IndexSizeError');
      return new ImageData(w, h);
    }
    getImageData(sx, sy, sw, sh) {
      if (!fin(sx, sy, sw, sh)) throw new TypeError('getImageData: non-finite argument');
      if (!sw || !sh) throw domError('The source width or height is 0.', 'IndexSizeError');
      if (sw < 0) { sx += sw; sw = -sw; } if (sh < 0) { sy += sh; sh = -sh; }
      sx = Math.floor(sx); sy = Math.floor(sy); sw = Math.ceil(sw); sh = Math.ceil(sh);
      this._prepare();
      const gl = this.gl, buf = new Uint8Array(sw * sh * 4);
      gl.readPixels(sx, this._h - sy - sh, sw, sh, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      const out = new ImageData(sw, sh), d = out.data, row = sw * 4;
      for (let y = 0; y < sh; y++) {
        const si = (sh - 1 - y) * row, di = y * row;
        for (let i = 0; i < row; i += 4) {
          const a = buf[si + i + 3];
          if (a === 0) continue;
          if (a === 255) { d[di + i] = buf[si + i]; d[di + i + 1] = buf[si + i + 1]; d[di + i + 2] = buf[si + i + 2]; }
          else {
            d[di + i] = Math.min(255, Math.round((buf[si + i] * 255) / a));
            d[di + i + 1] = Math.min(255, Math.round((buf[si + i + 1] * 255) / a));
            d[di + i + 2] = Math.min(255, Math.round((buf[si + i + 2] * 255) / a));
          }
          d[di + i + 3] = a;
        }
      }
      return out;
    }
    putImageData(img, dx, dy, dirtyX = 0, dirtyY = 0, dirtyW = img.width, dirtyH = img.height) {
      if (!fin(dx, dy, dirtyX, dirtyY, dirtyW, dirtyH)) return;
      this._prepare();
      if (dirtyW < 0) { dirtyX += dirtyW; dirtyW = -dirtyW; }
      if (dirtyH < 0) { dirtyY += dirtyH; dirtyH = -dirtyH; }
      if (dirtyX < 0) { dirtyW += dirtyX; dirtyX = 0; }
      if (dirtyY < 0) { dirtyH += dirtyY; dirtyY = 0; }
      if (dirtyX + dirtyW > img.width) dirtyW = img.width - dirtyX;
      if (dirtyY + dirtyH > img.height) dirtyH = img.height - dirtyY;
      if (dirtyW <= 0 || dirtyH <= 0) return;
      dx = Math.trunc(dx); dy = Math.trunc(dy);
      const gl = this.gl;
      if (!this._putTex) this._putTex = gl.createTexture();
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this._putTex);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.activeTexture(gl.TEXTURE0);
      // 像素操作以實體像素為單位；頂點座標是邏輯像素，所以除以 pixelRatio
      const pr = this._pr, x0 = (dx + dirtyX) / pr, y0 = (dy + dirtyY) / pr, x1 = x0 + dirtyW / pr, y1 = y0 + dirtyH / pr;
      const u0 = dirtyX / img.width, v0 = dirtyY / img.height, u1 = (dirtyX + dirtyW) / img.width, v1 = (dirtyY + dirtyH) / img.height;
      gl.disable(gl.STENCIL_TEST); gl.disable(gl.BLEND);
      const u = this._u;
      gl.uniform1i(u.u_type, 5); gl.uniform1f(u.u_alpha, 1); gl.uniform1i(u.u_useMask, 0); gl.uniform1i(u.u_quality, 1);
      this._drawVerts(new Float32Array([x0, y0, u0, v0, x1, y0, u1, v0, x1, y1, u1, v1, x0, y0, u0, v0, x1, y1, u1, v1, x0, y1, u0, v1]));
      gl.enable(gl.BLEND);
    }

    /* =================== 文字 =================== */
    measureText(text) { this._syncText(this._mctx); return this._mctx.measureText(String(text)); }
    fillText(text, x, y, maxWidth) { this._text(String(text), x, y, maxWidth, false); }
    strokeText(text, x, y, maxWidth) { this._text(String(text), x, y, maxWidth, true); }
    _text(text, x, y, maxWidth, stroke) {
      if (!fin(x, y) || (maxWidth !== undefined && !(maxWidth > 0))) return;
      text = text.replace(/[\t\n\f\r]/g, ' ');
      this._prepare(); this._syncClip();
      const s = this._state, mc = this._mctx;
      this._syncText(mc);
      const mt = mc.measureText(text);
      let left = -mt.actualBoundingBoxLeft, right = mt.actualBoundingBoxRight;
      const top = -mt.actualBoundingBoxAscent, bottom = mt.actualBoundingBoxDescent;
      if (!(right > left) || !(bottom > top)) return;
      const hs = maxWidth !== undefined && mt.width > maxWidth ? maxWidth / mt.width : 1;
      const pad = 2 + (stroke ? s.lineWidth * (s.lineJoin === 'miter' ? Math.max(1, s.miterLimit / 2) : 0.5) : 0);
      let sc = Math.max(0.05, maxScale(s.m) * this._pr);
      const bw = right - left + 2 * pad, bh = bottom - top + 2 * pad;
      sc = Math.min(sc, 4096 / bw, 4096 / bh);
      const key = [stroke ? 1 : 0, s.font, text, sc.toFixed(3), ...TEXT_PROPS.map((k) => s[k]),
        stroke ? [s.lineWidth, s.lineJoin, s.lineCap, s.miterLimit, s.lineDash.join(','), s.lineDashOffset].join() : ''].join('\u0001');
      let e = this._textCache.get(key);
      if (e) { this._textCache.delete(key); this._textCache.set(key, e); }
      else {
        const W = Math.max(1, Math.ceil(bw * sc)), H = Math.max(1, Math.ceil(bh * sc));
        const tc = this._tcanvas, t = this._tctx;
        tc.width = W; tc.height = H; // 重設尺寸同時清空畫布與狀態
        this._syncText(t);
        t.setTransform(sc, 0, 0, sc, (pad - left) * sc, (pad - top) * sc);
        if (stroke) {
          t.lineWidth = s.lineWidth; t.lineJoin = s.lineJoin; t.lineCap = s.lineCap; t.miterLimit = s.miterLimit;
          t.setLineDash(s.lineDash); t.lineDashOffset = s.lineDashOffset; t.strokeStyle = '#fff'; t.strokeText(text, 0, 0);
        } else { t.fillStyle = '#fff'; t.fillText(text, 0, 0); }
        const gl = this.gl, tex = gl.createTexture();
        gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, tc);
        this._texParams(); gl.activeTexture(gl.TEXTURE0);
        e = { tex, u1: (bw * sc) / W, v1: (bh * sc) / H };
        this._textCache.set(key, e);
        if (this._textCache.size > 256) {
          const [k0, e0] = this._textCache.entries().next().value;
          gl.deleteTexture(e0.tex); this._textCache.delete(k0);
        }
      }
      // 套用 maxWidth 水平壓縮（以錨點為中心）
      const qx = x + (left - pad) * hs, qw = bw * hs;
      const q = this._quad(qx, y + top - pad, qw, bh, 0, 0, e.u1, e.v1);
      this._drawConvex(q, stroke ? s.stroke : s.fill, { mask: e.tex });
    }
  }

  // 套用 filter / shadow 的繪圖方法：有效果時改走離屏圖層
  for (const name of ['fill', 'stroke', 'fillRect', 'strokeRect', 'fillText', 'strokeText', 'drawImage']) {
    const orig = WebGL2DRenderingContext.prototype[name];
    WebGL2DRenderingContext.prototype[name] = function (...a) {
      if (this._curFB) return orig.apply(this, a);
      const chain = this._effects();
      if (chain || BLEND_MODES[this._state.globalCompositeOperation]) return this._filtered(chain || [], () => orig.apply(this, a));
      return orig.apply(this, a);
    };
  }

  // 文字相關列舉屬性（textAlign、textBaseline…）的 getter / setter
  const TEXT_ENUMS = {
    textAlign: ['start', 'end', 'left', 'right', 'center'],
    textBaseline: ['top', 'hanging', 'middle', 'alphabetic', 'ideographic', 'bottom'],
    direction: ['ltr', 'rtl', 'inherit'],
    fontKerning: ['auto', 'normal', 'none'],
    fontStretch: ['ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'],
    fontVariantCaps: ['normal', 'small-caps', 'all-small-caps', 'petite-caps', 'all-petite-caps', 'unicase', 'titling-caps'],
    textRendering: ['auto', 'optimizeSpeed', 'optimizeLegibility', 'geometricPrecision'],
  };
  for (const [k, vals] of Object.entries(TEXT_ENUMS)) {
    Object.defineProperty(WebGL2DRenderingContext.prototype, k, {
      get() { return this._state[k]; },
      set(v) { if (vals.includes(v)) this._state[k] = v; },
      configurable: true, enumerable: true,
    });
  }
  for (const k of ['letterSpacing', 'wordSpacing']) {
    Object.defineProperty(WebGL2DRenderingContext.prototype, k, {
      get() { return this._state[k]; },
      set(v) {
        const c = this._mctx;
        if (!(k in c)) { this._state[k] = String(v); return; }
        c[k] = this._state[k]; c[k] = String(v); this._state[k] = c[k]; // 交由瀏覽器驗證 CSS 長度
      },
      configurable: true, enumerable: true,
    });
  }

  /* =====================================================================
   * 進入點
   * ===================================================================== */
  const ctxMap = new WeakMap();
  function getContext(canvas, opts) {
    let c = ctxMap.get(canvas);
    if (!c) { c = new WebGL2DRenderingContext(canvas, opts); ctxMap.set(canvas, c); }
    return c;
  }
  /**
   * 建立高解析度 context：canvas 的 backing store = CSS 尺寸 × devicePixelRatio，
   * 繪圖座標仍是 CSS 像素。
   *   opts.width / opts.height：CSS 尺寸（預設為 canvas 目前的 width / height）
   *   opts.pixelRatio：預設 devicePixelRatio
   *   opts.setStyle：是否設定 canvas.style 的寬高（預設 true）
   */
  function getContextHiDPI(canvas, opts = {}) {
    const pr = opts.pixelRatio > 0 ? +opts.pixelRatio : (typeof devicePixelRatio === 'number' ? devicePixelRatio : 1);
    const cw = opts.width !== undefined ? opts.width : canvas.width;
    const ch = opts.height !== undefined ? opts.height : canvas.height;
    if (opts.setStyle !== false && canvas.style) { canvas.style.width = cw + 'px'; canvas.style.height = ch + 'px'; }
    canvas.width = Math.round(cw * pr);
    canvas.height = Math.round(ch * pr);
    const ctx = getContext(canvas, Object.assign({}, opts, { pixelRatio: pr }));
    ctx.pixelRatio = pr;
    return ctx;
  }
  function install() {
    if (typeof HTMLCanvasElement === 'undefined') return;
    const proto = HTMLCanvasElement.prototype;
    if (proto.__webgl2d) return;
    const orig = proto.getContext;
    proto.getContext = function (type, opts) { return type === 'webgl-2d' ? getContext(this, opts) : orig.call(this, type, opts); };
    proto.__webgl2d = true;
  }

  return {
    version: '1.0.0',
    getContext, getContextHiDPI, install,
    WebGL2DRenderingContext, Path2D, CanvasGradient, CanvasPattern, ShaderFilter,
    _internal: { parseCSSFilter, parseColor, serializeColor, parseSVGPath, FlatPath, buildStroke, dashPolyline, windingAt },
  };
});
