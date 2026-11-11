/* VC Exit Engine: simulation core. Runs in a Web Worker for the page and under Node for tests. */
(function () {
  'use strict';

  const DT = 0.25;   // simulation step, years
  const EVERY = 2;   // a secondary sale can execute every EVERY steps (semi-annual)
  const FS = Array.from({ length: 21 }, (_, i) => i / 20);   // fractions of the stake sold now
  const YEAR_MS = 3.15576e10;

  const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
  const num = (x, d) => (x === null || x === undefined || x === '' || !isFinite(+x) ? d : +x);
  const logistic = (x) => 1 / (1 + Math.exp(-x));
  // standard normal CDF (Abramowitz-Stegun 7.1.26, error under 1.5e-7)
  function normCdf(x) {
    const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
    const e = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-x * x / 2);
    return x >= 0 ? 1 - e / 2 : e / 2;
  }
  // A typical secondary price is where deals clear: the best bid from the usual field of serious buyers (a Poisson count
  // averaging REF_BUYERS). Z_REF places that median best bid above one buyer's median, so the default field reproduces
  // the typical price and only a busier or quieter field than usual moves it.
  const REF_BUYERS = 2;
  const Z_REF = invNorm(1 + Math.log((1 + Math.exp(-REF_BUYERS)) / 2) / REF_BUYERS);
  function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function quantile(sorted, q) {
    if (!sorted.length) return NaN;
    const i = clamp(q * (sorted.length - 1), 0, sorted.length - 1), lo = Math.floor(i), hi = Math.ceil(i);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
  }
  // root of an increasing function on [lo, hi]
  function bisect(fn, lo, hi, it) { for (let k = 0; k < (it || 60); k++) { const m = (lo + hi) / 2; if (fn(m) < 0) lo = m; else hi = m; } return (lo + hi) / 2; }

  // xoshiro128** with Box-Muller normals
  function rng(seed) {
    let a = 0x9e3779b9 ^ seed, b = 0x243f6a88, c = 0xb7e15162 ^ Math.imul(seed, 31), d = 0x13198a2e;
    const rotl = (x, k) => (x << k) | (x >>> (32 - k));
    const next = () => {
      const r = Math.imul(rotl(Math.imul(b, 5), 7), 9) >>> 0, t = b << 9;
      c ^= a; d ^= b; b ^= c; a ^= d; c ^= t; d = rotl(d, 11);
      return r;
    };
    for (let i = 0; i < 20; i++) next();
    let spare = NaN;
    const u = () => (next() + 0.5) / 4294967296;
    const n = () => {
      if (spare === spare) { const v = spare; spare = NaN; return v; }
      const r = Math.sqrt(-2 * Math.log(u())), th = 6.283185307179586 * u();
      spare = r * Math.sin(th);
      return r * Math.cos(th);
    };
    return { u, n };
  }

  // Acklam's inverse normal CDF
  function invNorm(p) {
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    const tail = (q) => (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    if (p < 0.02425) return tail(Math.sqrt(-2 * Math.log(p)));
    if (p > 0.97575) return -tail(Math.sqrt(-2 * Math.log(1 - p)));
    const q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  // ---------- cap table and liquidation waterfall ----------

  function compileCap(rows) {
    const n = rows.length, F = () => new Float64Array(n), U = () => new Uint8Array(n);
    const ct = { n, rows, shares: F(), oip: F(), L: F(), cap: F(), rank: F(), cr: F(), thr: F(), ours: F(),
      part: U(), common: U(), out: F(), w: F(), conv: U(), capped: U(), fd: 0, oursConv: 0, oursShares: 0 };
    const pref = [];
    rows.forEach((r, i) => {
      ct.shares[i] = r.shares; ct.oip[i] = r.oip; ct.cr[i] = r.cr > 0 ? r.cr : 1; ct.rank[i] = r.rank; ct.ours[i] = r.ours;
      ct.L[i] = r.common ? 0 : r.shares * r.oip * r.prefMult;
      ct.common[i] = r.common || ct.L[i] <= 0 ? 1 : 0;   // a class with no preference behaves like common
      ct.part[i] = !ct.common[i] && r.participating ? 1 : 0;
      ct.cap[i] = ct.part[i] && r.capMult > 0 ? r.capMult * r.shares * r.oip : Infinity;
      ct.fd += r.shares * ct.cr[i]; ct.oursConv += r.ours * ct.cr[i]; ct.oursShares += r.ours;
      if (!ct.common[i]) pref.push(i);
      // conversion threshold: common price per share above which the class prefers to convert
      ct.thr[i] = ct.common[i] ? 0 : (ct.part[i] ? ct.cap[i] : ct.L[i]) / (r.shares * ct.cr[i]);
    });
    const ranks = [...new Set(pref.map((i) => ct.rank[i]))].sort((x, y) => x - y);
    ct.groups = ranks.map((k) => pref.filter((i) => ct.rank[i] === k));   // most senior first, pari passu within a group
    ct.order = pref.filter((i) => isFinite(ct.thr[i])).sort((x, y) => ct.thr[x] - ct.thr[y]);
    return ct;
  }

  // Pays preferences by seniority, then water-fills the rest across common-equivalent shares
  // (respecting participation caps). Fills ct.out, returns the price per common-equivalent share.
  function distribute(ct, E, conv) {
    const n = ct.n, out = ct.out, w = ct.w, capped = ct.capped;
    out.fill(0);
    let rem = E > 0 ? E : 0;
    for (const g of ct.groups) {
      if (rem <= 0) break;
      let Lg = 0;
      for (const i of g) if (!conv[i]) Lg += ct.L[i];
      if (Lg <= 0) continue;
      const pay = rem < Lg ? rem : Lg;
      for (const i of g) if (!conv[i]) out[i] = pay * ct.L[i] / Lg;
      rem -= pay;
    }
    if (rem <= 0) return 0;
    let S = 0;
    for (let i = 0; i < n; i++) {
      capped[i] = 0;
      w[i] = ct.common[i] || conv[i] || ct.part[i] ? ct.shares[i] * ct.cr[i] : 0;
      S += w[i];
    }
    let p = 0;
    for (let it = 0; it <= n; it++) {
      p = S > 0 ? rem / S : 0;
      let changed = false;
      for (let i = 0; i < n; i++) {
        if (capped[i] || !w[i] || conv[i] || !ct.part[i]) continue;
        const head = ct.cap[i] - out[i];
        if (p * w[i] > head) {
          const h = head > 0 ? head : 0;
          out[i] += h; rem -= h; S -= w[i]; capped[i] = 1; changed = true;
        }
      }
      if (!changed) break;
    }
    for (let i = 0; i < n; i++) if (w[i] && !capped[i]) out[i] += p * w[i];
    return p;
  }

  // Exit at equity value E. Each preferred class converts when that pays more; classes convert in order of
  // their threshold, which is the equilibrium because converting lowers the common price but never below
  // the converting class's own threshold. ipo = qualified IPO, every class converts. Returns our dollars.
  function settle(ct, E, ipo) {
    const conv = ct.conv;
    if (ipo) {
      conv.fill(1);
      distribute(ct, E, conv);
    } else {
      conv.fill(0);
      let p = distribute(ct, E, conv);
      for (const i of ct.order) {
        if (ct.thr[i] < p) { conv[i] = 1; p = distribute(ct, E, conv); } else break;
      }
    }
    let ours = 0;
    for (let i = 0; i < ct.n; i++) if (ct.ours[i] > 0) ours += ct.out[i] * ct.ours[i] / ct.shares[i];
    return ours;
  }

  // New financing at `price`. Down rounds trigger anti-dilution and come in senior to everything.
  function addRound(rows, ct, raise, price, down, k) {
    const next = rows.map((r) => Object.assign({}, r));
    if (down) {
      const A = ct.fd, issued = raise / price;
      for (const r of next) {
        if (r.common || r.ad === 'none') continue;
        const cp = r.oip / (r.cr || 1);
        if (!(price < cp)) continue;
        const cpNew = r.ad === 'fr' ? price : cp * (A + raise / cp) / (A + issued);   // broad-based weighted average
        r.cr = r.oip / cpNew;
      }
    }
    let senior = Infinity;
    for (const r of next) if (!r.common) senior = Math.min(senior, r.rank);
    if (!isFinite(senior)) senior = 1;
    next.push({ name: 'New round ' + k, shares: raise / price, oip: price, prefMult: 1, participating: false, capMult: 0,
      rank: down ? senior - 1 : senior, ad: 'wa', common: false, ours: 0, cr: 1 });
    return next;
  }

  // ---------- option pricing method (AICPA valuation guide) ----------

  const ZQ = (() => { const K = 800, z = new Float64Array(K); for (let k = 0; k < K; k++) z[k] = invNorm((k + 0.5) / K); return z; })();

  // Risk-neutral value of class idx per share (or of our whole position when idx < 0).
  function opmValue(ct, E0, sig, T, r, idx) {
    const mu = Math.log(E0) + (r - 0.5 * sig * sig) * T, s = sig * Math.sqrt(T);
    let acc = 0;
    for (let k = 0; k < ZQ.length; k++) {
      const ours = settle(ct, Math.exp(mu + s * ZQ[k]), false);
      acc += idx < 0 ? ours : ct.out[idx] / ct.shares[idx];
    }
    return Math.exp(-r * T) * acc / ZQ.length;
  }

  // 10-node Gauss-Hermite OPM value of our position: the fair value secondary buyers price off, cheap enough per path
  const GH = (() => {
    const x = [0.342901327223705, 1.036610829789514, 1.756683649299882, 2.53273167423279, 3.436159118837738];
    const w = [0.610862633735326, 0.240138611082315, 0.033874394455481, 0.001343645746781, 0.000007640432855];
    const z = [], wt = [];
    for (let i = 0; i < 5; i++) for (const s of [-1, 1]) { z.push(s * Math.SQRT2 * x[i]); wt.push(w[i] / Math.sqrt(Math.PI)); }
    return { z, wt };
  })();
  function fairQuick(ct, E, sig, T, r) {
    if (!(E > 0)) return 0;
    const mu = Math.log(E) + (r - 0.5 * sig * sig) * T, s = sig * Math.sqrt(T);
    let acc = 0;
    for (let k = 0; k < 10; k++) acc += GH.wt[k] * settle(ct, Math.exp(mu + s * GH.z[k]), false);
    return Math.exp(-r * T) * acc;
  }

  // Equity value at which the last round's class is worth exactly its issue price.
  function backsolve(ct, idx, price, sig, T, r) {
    const head = price * ct.fd;
    return Math.exp(bisect((x) => opmValue(ct, Math.exp(x), sig, T, r, idx) - price, Math.log(head * 1e-4), Math.log(head * 4), 50));
  }

  // ---------- inputs ----------

  function yearsSince(iso, fallback) {
    const t = iso ? Date.parse(iso) : NaN;
    return isFinite(t) ? Math.max(0, (Date.now() - t) / YEAR_MS) : fallback;
  }

  function normalize(raw) {
    const r = raw || {}, f = r.family || {}, c = r.company || {}, h = r.holding || {}, m = r.market || {}, s = r.sim || {};
    const F = {
      wealth: Math.max(1, num(f.wealth, 250e6)),
      gamma: clamp(num(f.gamma, 3), 0, 15),
      ambiguity: clamp(num(f.ambiguity, 1), 0, 10),
      hurdle: clamp(num(f.hurdle, 0.15), 0, 0.6),
      riskFree: clamp(num(f.riskFree, 0.04), -0.02, 0.15),
      portfolioReturn: clamp(num(f.portfolioReturn, 0.07), -0.05, 0.3),
      marketVol: clamp(num(f.marketVol, 0.16), 0.01, 0.6),
      equityWeight: clamp(num(f.equityWeight, 0.7), 0, 1.5),
      taxFed: clamp(num(f.taxFed, 0.238), 0, 0.6),
      taxState: clamp(num(f.taxState, 0), 0, 0.2),
      stateConformsQsbs: f.stateConformsQsbs !== false,
      qsbsAllowed: f.qsbsAllowed !== false,
      lossesUsable: f.lossesUsable !== false,
      horizon: clamp(Math.round(num(f.horizon, 8) * 2) / 2, 2, 12),
      window: clamp(num(f.window, 0), -2, 2),
      // the bank's engagement terms: fee and timing are every stake's defaults; the minimum and retainer are shared by
      // the whole package, so only the portfolio plan charges them
      bankFee: clamp(num(f.bankFee, 0.03), 0, 0.2),
      bankMonths: clamp(num(f.bankMonths, 0), 0, 24),
      bankMinFee: Math.max(0, num(f.bankMinFee, 0)),
      bankUpfront: Math.max(0, num(f.bankUpfront, 0)),
    };
    const revenue = Math.max(0, num(c.revenue, 0)), burn = Math.max(0, num(c.monthlyBurn, 0));
    const C = {
      name: String(c.name || 'Company').slice(0, 80),
      revenue,
      growth: clamp(num(c.growth, 0.3), -0.6, 3),
      fcfMargin: clamp(num(c.fcfMargin, revenue > 0 ? -12 * burn / revenue : 0), -5, 0.6),
      monthlyBurn: burn,
      cash: Math.max(0, num(c.cash, 0)),
      debt: Math.max(0, num(c.debt, 0)),
      lastRoundDate: c.lastRoundDate || null,
      indexMove: clamp(num(c.indexMove, 0), -0.95, 5),
      compMultiple: clamp(num(c.compMultiple, 6), 0.1, 100),
      privateDiscount: clamp(num(c.privateDiscount, 0.2), 0, 0.9),
      longRunGrowth: clamp(num(c.longRunGrowth, 0.05), -0.2, 0.3),
      targetMargin: clamp(num(c.targetMargin, 0.15), -0.5, 0.5),
      valueGrowth: clamp(num(c.valueGrowth, 0.1), -0.5, 1),
      vol: clamp(num(c.vol, 0.9), 0.1, 2.5),
      disruption: clamp(num(c.disruption, 0.03), 0, 1),
      maHazard: clamp(num(c.maHazard, 0.08), 0, 1),
      ipoHazard: clamp(num(c.ipoHazard, 0.3), 0, 2),
      ipoRevenue: Math.max(1e6, num(c.ipoRevenue, 200e6)),
      exitYears: clamp(num(c.exitYears, 3), 0.5, 10),
    };
    const cap = (Array.isArray(r.cap) ? r.cap : []).map((x, i) => ({
      name: String(x.name || 'Class ' + (i + 1)).slice(0, 40),
      shares: Math.max(0, num(x.shares, 0)),
      oip: Math.max(0, num(x.oip, 0)),
      prefMult: clamp(num(x.prefMult, 1), 0, 5),
      participating: !!x.participating,
      capMult: clamp(num(x.capMult, 0), 0, 10),
      rank: clamp(Math.round(num(x.rank, 1)), 0, 20),
      ad: x.ad === 'fr' || x.ad === 'none' ? x.ad : 'wa',
      common: !!x.common,
      ours: Math.max(0, num(x.ours, 0)),
      cr: 1,
    })).filter((x) => x.shares > 0);
    for (const x of cap) x.ours = Math.min(x.ours, x.shares);
    const H = {
      costBasis: Math.max(0, num(h.costBasis, 0)),
      holdYears: yearsSince(h.acquired, num(h.holdYears, 6)),
      qsbs: !!h.qsbs,
      carry: clamp(num(h.carry, 0), 0, 0.5),
    };
    const bids = (Array.isArray(m.bids) ? m.bids : []).map((b) => Math.max(0, num(b && b.price, 0)));
    // three or more bids show how far apart buyers' prices land; otherwise the spread is a placeholder
    const lb = bids.filter((b) => b > 0).map(Math.log), lbm = lb.length ? lb.reduce((a, x) => a + x, 0) / lb.length : 0;
    const bidSd = lb.length >= 3 ? Math.sqrt(lb.reduce((a, x) => a + (x - lbm) ** 2, 0) / (lb.length - 1)) : NaN;
    const M = {
      bid: bids.length ? Math.max(...bids) : 0,
      fee: clamp(num(m.fee, F.bankFee), 0, 0.2),
      minFee: Math.max(0, num(m.minFee, 0)),   // a minimum fee this stake bears on its own
      upfront: Math.max(0, num(m.upfront, 0)),   // retainer and legal this stake bears, paid even if no sale closes
      lag: clamp(num(m.monthsToClose, F.bankMonths), 0, 24) / 12,   // years from deciding to sell to cash in hand
      buyers: clamp(num(m.buyers, REF_BUYERS), 0, 50),   // serious buyers a marketed sale draws: a placeholder until the bank reports
      spread: clamp(num(m.bidSpread, isFinite(bidSd) ? Math.max(0.05, bidSd) : 0.25), 0.02, 1.5),   // log spread of their prices
      spreadFromBids: num(m.bidSpread, null) === null && isFinite(bidSd),
      rofr: !!m.rofr, consent: !!m.consent, blocked: !!m.blocked,
      noBidDiscount: clamp(num(m.noBidDiscount, 0.3), 0, 0.9),
      pClose: m.pClose === '' || m.pClose == null ? null : clamp(+m.pClose, 0, 1),
    };
    if (M.pClose === null || !isFinite(M.pClose)) M.pClose = M.blocked ? 0 : (M.consent ? 0.6 : 0.95) * (M.rofr ? 0.95 : 1);
    const S = { paths: clamp(Math.round(num(s.paths, 12000)), 500, 60000), bundles: clamp(Math.round(num(s.bundles, 16)), 1, 32), seed: Math.round(num(s.seed, 20261029)) };
    return { F, C, H, M, cap, S };
  }

  // Private-market EV/revenue at quality q = growth + ½·margin, anchored to the sector multiple at q = 0.25
  // (25% growth at breakeven). Slope k near the anchor; tanh caps the premium or discount at about 3.5×.
  const multipleAt = (C, q, k) => C.compMultiple * (1 - C.privateDiscount) * Math.exp((k / 2) * Math.tanh(2 * (q - 0.25)));

  // ---------- today's equity value ----------
  // Last round -> OPM backsolve -> rolled by the comparable index -> blended with a revenue-multiple value,
  // the blend shifting to the multiple as the round goes stale (IPEV calibration).
  function markBridge(I, ct) {
    const { C, F } = I;
    let last = -1;
    for (let i = 0; i < ct.n; i++) if (!ct.common[i]) last = i;   // the last preferred row is the most recent round
    const pLast = last >= 0 ? ct.oip[last] : 0;
    const post = pLast * ct.fd;
    const ageY = yearsSince(C.lastRoundDate, 2);
    const sig = C.revenue > 0 ? 0.6 : Math.min(C.vol, 1.2);
    const eOpm = pLast > 0 ? backsolve(ct, last, pLast, sig, C.exitYears, F.riskFree) : NaN;
    const eRoll = isFinite(eOpm) ? eOpm * (1 + C.indexMove) : NaN;
    const eMult = C.revenue > 0 ? C.revenue * multipleAt(C, C.growth + 0.5 * C.fcfMargin, 2.5) + C.cash - C.debt : NaN;
    if (!isFinite(eRoll) && !isFinite(eMult)) throw new Error('To value the company, enter revenue or a preferred class with an issue price.');
    const w = !isFinite(eRoll) ? 0 : !isFinite(eMult) ? 1 : Math.exp(-ageY / 1.5);
    const floor = 0.01 * Math.max(isFinite(eRoll) ? eRoll : 0, isFinite(eMult) ? eMult : 0, 1);
    const e0 = w >= 1 ? eRoll : w <= 0 ? Math.max(eMult, floor)
      : Math.exp(w * Math.log(Math.max(eRoll, floor)) + (1 - w) * Math.log(Math.max(eMult, floor)));
    return { post, eOpm, eRoll, eMult, wRound: w, e0, ageY, last, pLast, sig, lastClass: last >= 0 ? ct.rows[last].name : null };
  }

  // After SPV carry and capital-gains tax (QSBS exclusion once held 5 years; usable losses give a credit).
  function makeTax(I) {
    const { F, H } = I;
    const qsbs = H.qsbs && F.qsbsAllowed, qcap = Math.max(10e6, 10 * H.costBasis), rate = F.taxFed + F.taxState;
    return (gross, t) => {
      let x = gross;
      if (H.carry > 0 && x > H.costBasis) x -= H.carry * (x - H.costBasis);
      const gain = x - H.costBasis;
      if (gain <= 0) return x - (F.lossesUsable ? gain * rate : 0);
      const excl = qsbs && H.holdYears + t >= 5 ? Math.min(gain, qcap) : 0;
      return x - (gain - excl) * F.taxFed - (F.stateConformsQsbs ? gain - excl : gain) * F.taxState;
    };
  }

  // ---------- Monte Carlo ----------

  function bundleParams(C, B, seed) {
    const base = { gInf: C.longRunGrowth, lamG: 0.35, sigG: 0.1, sigR: 0.12, mT: C.targetMargin, lamM: 0.25, k: 2.5, compAdj: 1,
      ipo: C.ipoHazard, ma: C.maHazard, jump: C.disruption, fin0: 0.8, muV: C.valueGrowth, sigV: C.vol };
    if (B === 1) return [base];
    const R = rng(seed ^ 0x5bd1e995), ln = (s) => Math.exp(s * R.n() - 0.5 * s * s);   // mean-one lognormal
    const out = [];
    for (let b = 0; b < B; b++) {
      out.push({ gInf: base.gInf + 0.02 * R.n(), lamG: base.lamG * ln(0.3), sigG: base.sigG, sigR: base.sigR * ln(0.2),
        mT: base.mT + 0.04 * R.n(), lamM: base.lamM * ln(0.3), k: base.k * ln(0.25), compAdj: ln(0.2),
        ipo: base.ipo * ln(0.4), ma: base.ma * ln(0.4), jump: base.jump * ln(0.5), fin0: base.fin0 + 0.4 * R.n(),
        muV: base.muV + 0.05 * R.n(), sigV: base.sigV * ln(0.15) });
    }
    return out;
  }

  function simulate(I, ct0, mk, opt, dBase, pClose) {
    const { F, C } = I;
    const N = opt.paths, B = opt.bundles, per = Math.ceil(N / B);
    const steps = Math.round(F.horizon / DT), D = steps / EVERY - 1;   // sale dates strictly before the horizon
    const sq = Math.sqrt(DT), sm = F.marketVol, we = F.equityWeight;
    // cash (freed by a sale, or paid out at exit) joins the family portfolio: its expected return, market exposure `we`
    const redeploy = (F.portfolioReturn - 0.5 * we * we * sm * sm) * DT;
    const revMode = C.revenue > 0;
    const g0 = Math.log(1 + C.growth), m0 = C.fcfMargin, r40_0 = revMode ? C.growth + m0 : 0;
    const mult0 = Math.max(0.05, (mk.e0 - C.cash + C.debt) / Math.max(C.revenue, 1));
    const lnV0 = Math.log(Math.max(1, mk.e0 - C.cash + C.debt));
    const lnIpo = Math.log(C.ipoRevenue), lnBig = Math.log(3e9);
    const P = bundleParams(C, B, opt.seed);
    const sim = { N, B, steps, D, bundleOf: new Uint8Array(N), type: new Uint8Array(N), event: new Float64Array(N),
      tau: new Float64Array(N), gross: new Float64Array(N), rounds: new Uint8Array(N), closes: new Uint8Array(N),
      cum: new Float64Array(N * (steps + 1)), bid: new Float32Array(N * D), feat: new Float32Array(N * D * 4) };
    const fair = (ct, E) => fairQuick(ct, E, mk.sig, C.exitYears, F.riskFree);
    const fair0 = Math.max(fair(ct0, mk.e0), 1);
    const discount = (w, r40, R) => clamp(dBase - 0.08 * w - 0.25 * (r40 - r40_0) + 0.05 * R.n(), 0, 0.85);

    for (let p = 0; p < N; p++) {
      const b = Math.min(B - 1, (p / per) | 0), Q = P[b];
      const R = rng((opt.seed ^ Math.imul(p + 1, 0x9e3779b1)) >>> 0);   // one stream per path keeps runs comparable
      sim.bundleOf[p] = b;
      sim.closes[p] = R.u() < pClose ? 1 : 0;
      let lnRev = Math.log(Math.max(C.revenue, 1)), g = g0, m = m0, cash = C.cash, w = F.window, z = 0, cum = 0;
      let prem = Math.log(mult0 / (multipleAt(C, C.growth + 0.5 * m0, Q.k) * Q.compAdj)), lnV = lnV0, burn = C.monthlyBurn * 12;
      let rows = I.cap, ct = ct0, lastPrice = mk.pLast > 0 ? mk.pLast : mk.e0 / ct0.fd, eLast = mk.e0, rounds = 0;
      let type = 0, event = F.horizon, tau = F.horizon, gross = 0, E = mk.e0, d = 0;
      const base = p * (steps + 1);
      for (let s = 1; s <= steps; s++) {
        const t = s * DT, em = R.n();
        cum += redeploy + we * sm * sq * em;
        sim.cum[base + s] = cum;
        z += sm * sq * em;   // market surprise; valuation multiples load on it
        w += -0.6 * w * DT + 0.8 * sq * (0.6 * em + 0.8 * R.n());   // exit-window index, mean-reverting, tied to markets
        let ev, r40;
        if (revMode) {
          const mPrev = m;
          m += Q.lamM * (Q.mT - m) * DT + 0.05 * sq * R.n();
          g += Q.lamG * (Q.gInf - g) * DT + Q.sigG * sq * R.n() - 0.35 * (m - mPrev);   // cutting burn costs growth
          lnRev += g * DT + Q.sigR * sq * (0.3 * em + 0.954 * R.n());
          prem += -0.3 * prem * DT + 0.15 * sq * R.n();
          if (R.u() < Q.jump * DT) { g -= 0.25; prem += Math.log(0.6); }   // disruption shock
          const rev = Math.exp(lnRev);
          cash += m * rev * DT;
          r40 = Math.expm1(g) + m;
          // public markets pay roughly twice as much for growth as for margin
          ev = rev * multipleAt(C, Math.expm1(g) + 0.5 * m, Q.k) * Q.compAdj * Math.exp(1.5 * z + prem);
          burn = m < 0 ? -m * rev : 0;
        } else {
          lnV += (Q.muV - 0.5 * Q.sigV * Q.sigV) * DT + Q.sigV * sq * (0.5 * em + 0.866 * R.n());
          if (R.u() < Q.jump * DT) lnV += Math.log(0.4);
          burn *= Math.exp(-0.15 * DT);
          cash -= burn * DT;
          r40 = 0;
          ev = Math.exp(lnV);
        }
        E = ev + cash - C.debt;

        // runway under 9 months: raise (dilution, anti-dilution, senior paper on a down round) or fail
        if (burn > 0 && cash < 0.75 * burn) {
          const pre = Math.max(E, 1);
          if (R.u() < logistic(Q.fin0 + 4 * r40 + 0.8 * w + 0.8 * clamp(Math.log(pre / eLast), -2, 1))) {
            const raise = 2 * burn, price = pre / ct.fd;
            rows = addRound(rows, ct, raise, price, price < lastPrice, ++rounds);
            ct = compileCap(rows);
            cash += raise; E += raise; lastPrice = price; eLast = E;
          } else if (cash <= 0) {
            type = 3; event = tau = t;
            gross = settle(ct, ev * (0.1 + 0.3 * R.u()) - C.debt, false);   // fire sale of the assets
            break;
          }
        }

        // competing exit hazards
        const sIpo = revMode ? logistic((lnRev - lnIpo) / 0.35) * logistic((r40 - 0.15) / 0.08) : logistic((Math.log(Math.max(E, 1)) - lnBig) / 0.4);
        const pI = 1 - Math.exp(-Q.ipo * sIpo * Math.exp(w) * DT);
        const pM = 1 - Math.exp(-Q.ma * Math.exp(0.35 * w) * (0.6 + 0.8 * logistic((r40 - 0.1) / 0.1)) * DT);
        const u = R.u();
        if (u < pI) {
          type = 1; event = t; tau = Math.min(t + 0.5, F.horizon);
          gross = settle(ct, E, true) * Math.exp(0.3536 * R.n() - 0.0625);   // 6-month lock-up at 50% vol
          break;
        }
        if (u < pI + pM) {
          type = 2; event = tau = t;
          gross = settle(ct, ev * Math.exp(0.0953 + 0.25 * R.n() - 0.03125) + cash - C.debt, false);   // ~10% control premium
          break;
        }

        if (s % EVERY === 0 && s < steps) {
          const k = p * D + d;
          sim.bid[k] = fair(ct, E) * (1 - discount(w, r40, R));
          sim.feat[4 * k] = Math.log(Math.max(sim.bid[k], 1e-6 * fair0) / fair0);
          sim.feat[4 * k + 1] = r40;
          sim.feat[4 * k + 2] = w;
          sim.feat[4 * k + 3] = burn > 0 ? clamp(cash / burn, 0, 5) : 5;
          d++;
        }
        if (s === steps) gross = fair(ct, E) * (1 - discount(w, r40, R));   // stake still private: sold at the horizon
      }
      sim.type[p] = type; sim.event[p] = event; sim.tau[p] = tau; sim.gross[p] = gross; sim.rounds[p] = rounds;
    }
    return sim;
  }

  function solve(A, b, K) {
    const M = Array.from({ length: K }, (_, i) => Array.from({ length: K + 1 }, (_, j) => (j < K ? A[i * K + j] : b[i])));
    for (let c = 0; c < K; c++) {
      let piv = c;
      for (let r = c + 1; r < K; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      const dv = M[c][c];
      if (Math.abs(dv) < 1e-300) continue;
      for (let r = 0; r < K; r++) {
        if (r === c) continue;
        const f = M[r][c] / dv;
        if (f) for (let j = c; j <= K; j++) M[r][j] -= f * M[c][j];
      }
    }
    return M.map((row, i) => (Math.abs(row[i]) < 1e-300 ? 0 : row[K] / row[i]));
  }

  // Longstaff-Schwartz: at each sale date, sell when today's after-tax bid beats the regressed value of waiting.
  // Cross-fitted: odd paths follow the rule fitted on even paths and the reverse, so no path's own future leaks
  // into its decision (fitting in-sample overstates the value of waiting). inSample is kept for validation.
  function optimalStopping(sim, cf, cash, when, how, sell, lag, DF, inSample) {
    const { N, D } = sim, K = 9, phi = new Float64Array(K);
    const A = [new Float64Array(K * K), new Float64Array(K * K)], y = [new Float64Array(K), new Float64Array(K)], n = [0, 0];
    const basis = (k) => {
      const x = sim.feat[4 * k], r = sim.feat[4 * k + 1], w = sim.feat[4 * k + 2], rw = sim.feat[4 * k + 3];
      phi[0] = 1; phi[1] = x; phi[2] = x * x; phi[3] = x * x * x; phi[4] = r; phi[5] = w; phi[6] = rw; phi[7] = x * w; phi[8] = x * r;
    };
    const fit = (Am, ym) => {
      const M = Float64Array.from(Am);
      for (let i = 0; i < K; i++) { for (let j = i + 1; j < K; j++) M[i * K + j] = M[j * K + i]; M[i * K + i] += 1e-8 * M[0] + 1e-9; }
      return solve(M, ym, K);
    };
    for (let d = D - 1; d >= 0; d--) {
      const t = (d + 1) * EVERY * DT;
      for (let h = 0; h < 2; h++) { A[h].fill(0); y[h].fill(0); n[h] = 0; }
      for (let p = 0; p < N; p++) {
        if (sim.event[p] <= t) continue;
        const h = p & 1, Ah = A[h], yh = y[h];
        basis(p * D + d);
        for (let i = 0; i < K; i++) { yh[i] += phi[i] * cf[p]; for (let j = 0; j <= i; j++) Ah[i * K + j] += phi[i] * phi[j]; }
        n[h]++;
      }
      if (n[0] < 50 || n[1] < 50) continue;
      let beta;
      if (inSample) { const b = fit(A[0].map((v, i) => v + A[1][i]), y[0].map((v, i) => v + y[1][i])); beta = [b, b]; }
      else beta = [fit(A[1], y[1]), fit(A[0], y[0])];
      for (let p = 0; p < N; p++) {
        if (sim.event[p] <= t) continue;
        const k = p * D + d, b = beta[p & 1];
        basis(k);
        let cont = 0;
        for (let i = 0; i < K; i++) cont += b[i] * phi[i];
        const ex = sell(sim.bid[k], t);
        if (ex > cont) { cf[p] = ex; when[p] = t + lag; cash[p] = ex / DF(when[p]); how[p] = 4; }
      }
    }
  }

  // Expected CRRA utility of family wealth at the horizon. Cash compounds in the family's public portfolio along
  // each path's own market draws; parameter bundles are aggregated with smooth ambiguity aversion (KMM).
  function utilityLens(I, sim, cash, when, fv) {
    const F = I.F, N = sim.N, st = sim.steps, B = sim.B, gam = F.gamma, lagS = Math.min(st, Math.round(I.M.lag / DT));
    const g0H = new Float64Array(N), gLag = new Float64Array(N), holdH = new Float64Array(N);
    let Gbar = 0;
    for (let p = 0; p < N; p++) {
      const base = p * (st + 1), end = sim.cum[base + st];
      g0H[p] = Math.exp(end);
      gLag[p] = Math.exp(end - sim.cum[base + lagS]);   // sale cash lands after the closing lag
      holdH[p] = cash[p] * Math.exp(end - sim.cum[base + Math.min(st, Math.round(when[p] / DT))]);
      Gbar += g0H[p];
    }
    Gbar /= N;
    const scale = F.wealth * Gbar, rest = Math.max(0, F.wealth - fv);
    const alpha = F.ambiguity / Math.max(1e-6, fv / F.wealth);   // ambiguity aversion per unit of position size
    // sellCash[j]: after-tax cash from selling fs[j] of the stake; upfront costs are paid on any attempt, closed or not
    function curve(sellCash, upfront, fs) {
      const nf = fs.length, acc = new Float64Array(B * nf), cnt = new Float64Array(B);
      for (let p = 0; p < N; p++) {
        const b = sim.bundleOf[p], base = rest * g0H[p], cl = sim.closes[p];
        cnt[b]++;
        for (let j = 0; j < nf; j++) {
          const W = base - (fs[j] > 0 ? upfront * g0H[p] : 0) + (cl ? sellCash[j] * gLag[p] + (1 - fs[j]) * holdH[p] : holdH[p]);
          const wn = Math.max(W / scale, 1e-9);
          acc[b * nf + j] += gam === 1 ? Math.log(wn) : Math.pow(wn, 1 - gam);
        }
      }
      // certainty-equivalent gain of selling fs[j] versus holding, per bundle; the ambiguity penalty acts on the
      // gain, so market sampling noise shared by every fraction in a bundle cancels out
      const ceOf = (b, j) => { const mu = acc[b * nf + j] / cnt[b]; return gam === 1 ? Math.exp(mu) : Math.pow(mu, 1 / (1 - gam)); };
      const gain = new Float64Array(nf), xs = new Float64Array(B), ce0 = Array.from({ length: B }, (_, b) => ceOf(b, 0));
      for (let j = 1; j < nf; j++) {
        let mx = -Infinity, sum = 0;
        for (let b = 0; b < B; b++) { xs[b] = ceOf(b, j) - ce0[b]; sum += xs[b]; }
        if (!(alpha > 0) || B === 1) { gain[j] = sum / B; continue; }
        for (let b = 0; b < B; b++) { xs[b] *= -alpha; if (xs[b] > mx) mx = xs[b]; }
        let s = 0;
        for (let b = 0; b < B; b++) s += Math.exp(xs[b] - mx);
        gain[j] = -(mx + Math.log(s / B)) / alpha;
      }
      return gain;
    }
    return { curve };
  }

  // ---------- the decision ----------

  function run(raw, opts) {
    const I = normalize(raw);
    const S = Object.assign({}, I.S, opts || {});
    if (!I.cap.length) throw new Error('Add the cap table: at least one share class with a share count.');
    const ct0 = compileCap(I.cap);
    if (ct0.oursShares <= 0) throw new Error('Enter how many shares you hold in at least one class.');
    const { F, H, M } = I, warnings = [];
    const mk = markBridge(I, ct0);
    const liq0 = settle(ct0, mk.e0, false);
    const perClass = Array.from(ct0.out);
    const fvOpm = opmValue(ct0, mk.e0, mk.sig, I.C.exitYears, F.riskFree, -1);
    const sh = ct0.oursShares, hasBid = M.bid > 0;
    const typical = fvOpm * (1 - clamp(M.noBidDiscount - 0.08 * F.window, 0, 0.85));   // a buyer's typical price; today's window moves it
    const S0gross = hasBid ? M.bid * sh : typical;
    const net = makeTax(I), fee = M.fee, DF = (t) => Math.pow(1 + F.hurdle, -t);
    // Every secondary sale pays the fee with the bank's minimum (dearer on small sales) and upfront costs, and its cash
    // lands after the closing lag. cashOf: after-tax cash from selling fraction f at whole-stake price g, decided at t.
    // Later sales are assumed to close, as elsewhere in the simulation; today's pays its upfront costs once per
    // expected attempt, since a failed sale pays them too.
    const feeOf = (g) => Math.max(fee * g, M.minFee), tries = M.upfront / Math.max(M.pClose, 0.05);
    const cashOf = (g, t, f = 1) => f * net(g - feeOf(f * g) / f, t + M.lag);
    const sell = (g, t) => cashOf(g, t) * DF(t + M.lag) - M.upfront * DF(t);
    const today = (g, f = 1) => cashOf(g, 0, f) * DF(M.lag) - tries;

    // future resale prices use the name's typical discount, not today's bid, so the price ladder never moves with the offer
    const sim = simulate(I, ct0, mk, S, M.noBidDiscount, M.pClose);   // the typical discount is a neutral-window figure
    const N = sim.N;
    const cf = new Float64Array(N), cash = new Float64Array(N), when = new Float64Array(N), how = new Uint8Array(N);
    for (let p = 0; p < N; p++) {
      const sold = sim.type[p] === 0;   // still private at the horizon: sold like any secondary, unless costs exceed the price
      when[p] = sim.tau[p] + (sold ? M.lag : 0); how[p] = sim.type[p];
      cf[p] = sold ? Math.max(0, sell(sim.gross[p], sim.tau[p])) : net(sim.gross[p], when[p]) * DF(when[p]);
      cash[p] = cf[p] / DF(when[p]);
    }
    const cfHold = Float64Array.from(cf), holdPV = mean(cf);
    optimalStopping(sim, cf, cash, when, how, sell, M.lag, DF, S.lsmInSample);
    const vWait = mean(cf);
    let sd = 0;
    for (let p = 0; p < N; p++) sd += (cf[p] - vWait) ** 2;
    const se = Math.sqrt(sd / (N - 1) / N) / vWait;
    const S0net = today(S0gross);
    const sr = vWait > 0 ? S0net / vWait : S0net > 0 ? Infinity : 0;

    // Price test under parameter uncertainty: hold value per assumption bundle. Selling the minimax-regret
    // fraction (Manski) hedges between pessimistic and optimistic assumptions. The P10-P90 band comes from the spread
    // of bundle values with simulation noise subtracted, so the sale size reflects model uncertainty, not dice rolls.
    const vb = new Float64Array(sim.B), v2 = new Float64Array(sim.B), nb = new Float64Array(sim.B);
    for (let p = 0; p < N; p++) { const b = sim.bundleOf[p]; vb[b] += cf[p]; v2[b] += cf[p] * cf[p]; nb[b]++; }
    const vB = Array.from(vb, (v, b) => v / nb[b]);
    const noiseVar = mean(Array.from(vb, (v, b) => Math.max(0, v2[b] / nb[b] - (v / nb[b]) ** 2) / nb[b]));
    const paramSd = sim.B > 1 ? Math.sqrt(Math.max(0, vB.reduce((s, v) => s + (v - mean(vB)) ** 2, 0) / (sim.B - 1) - noiseVar)) : 0;
    const agree = vB.filter((v) => (sr >= 1 ? S0net >= v : S0net < v)).length / sim.B;
    const vLo = Math.max(0, vWait - 1.2816 * paramSd), vHi = vWait + 1.2816 * paramSd;
    // the price test's sale fraction at whole-stake price g: each fraction's worst regret is at P10 or P90, and a minimum
    // fee or upfront costs make small sales dearer, so each fraction is tested
    const hedge = (g) => {
      const sellF = FS.map((f) => (f > 0 ? today(g, f) : 0));
      const pay = (j, v) => sellF[j] + (1 - FS[j]) * v;
      const regret = (j) => Math.max(...[vLo, vHi].map((v) => Math.max(...FS.map((_, i) => pay(i, v))) - pay(j, v)));
      let jH = 0;
      for (let j = 1; j < FS.length; j++) if (regret(j) < regret(jH) - 1e-9 * Math.abs(vWait)) jH = j;
      return FS[jH];
    };
    const fHedge = hedge(S0gross);

    // Concentration test: family utility can add a trim, never block a sale that passes the price test
    const U = utilityLens(I, sim, cash, when, fvOpm);
    const ce = U.curve(FS.map((f) => (f > 0 ? cashOf(S0gross, 0, f) : 0)), M.upfront, FS);
    let jStar = 0;
    for (let j = 1; j < FS.length; j++) if (ce[j] > ce[jStar] + 1e-12) jStar = j;
    const fUtil = FS[jStar];
    const ceGain = Array.from(ce, (x) => x * F.wealth);
    const fStar = M.pClose < 0.05 ? 0 : Math.max(fHedge, fUtil);

    // price ladder, the price test inverted: below walkAway it sells nothing, at fair it is even, from sellAll it sells all
    const netAt = (x) => today(x * sh);
    const hi = 20 * Math.max(liq0, fvOpm, S0gross, M.minFee + tries) / sh;
    const priceFor = (v) => bisect((x) => netAt(x) - v, 0, hi, 60);
    const from = (ok) => bisect((x) => (ok(hedge(x * sh)) ? 1 : -1), 0, hi, 60);
    const ladder = { walkAway: from((f) => f > 0), fair: priceFor(vWait), sellAll: from((f) => f >= 1) };

    // A marketed sale: serious buyers come as a Poisson count, each bidding lognormally around one buyer's median (set so
    // the usual field clears at the typical price); the best bid, or the bid in hand if higher, sells when it reaches the
    // reserve (the walk-away price). P(best < x) = exp(-buyers * P(one bid >= x)), in whole-stake dollars. Upfront costs
    // are paid once, whatever happens.
    // shortcut: the sale policy at each price still charges upfront costs per attempt, slightly conservative for a stake
    // with its own retainer; split the policy if per-stake retainers become common.
    const one = typical * Math.exp(-M.spread * Z_REF);
    const under = (x) => (x <= M.bid * sh ? 0 : one > 0 ? Math.exp(-M.buyers * (1 - normCdf(Math.log(x / one) / M.spread))) : 1);
    const R = Math.max(ladder.walkAway * sh, 1e-6 * Math.max(typical, S0gross, 1));
    const proc = { reserve: R / sh, typical: typical / sh, buyers: M.buyers, spread: M.spread, spreadFromBids: M.spreadFromBids,
      noOffers: hasBid ? 0 : Math.exp(-M.buyers), pSale: M.pClose * (1 - under(R)), gain: -M.upfront, gross: 0, cash: 0, price: NaN };
    const K = 160, dl = (Math.log(Math.max(R, M.bid * sh, typical) / R) + 5 * M.spread) / K;
    let px = 0;
    for (let k = 0, lo = R, cLo = under(R); k <= K; k++) {
      const hiX = k < K ? R * Math.exp((k + 1) * dl) : Infinity, cHi = k < K ? under(hiX) : 1;
      const x = k < K ? Math.sqrt(lo * hiX) : lo, w = M.pClose * (cHi - cLo), f = w > 0 ? hedge(x) : 0;
      if (f > 0) { const c = cashOf(x, 0, f); proc.gain += w * (c * DF(M.lag) - f * vWait); proc.gross += w * f * x; proc.cash += w * c; px += w * x; }
      lo = hiX; cLo = cHi;
    }
    proc.price = proc.pSale > 0 ? px / proc.pSale / sh : NaN;

    const pv = (r) => { let s = 0; for (let p = 0; p < N; p++) s += cash[p] * Math.pow(1 + r, -when[p]); return s / N; };
    const irr = S0net > 0 ? bisect((r) => S0net - pv(r), -0.95, 5, 60) : NaN;

    // outcome statistics (hold to exit, then under the optimal resale policy)
    const mix = [0, 0, 0, 0, 0], pol = [0, 0, 0, 0, 0], mult = new Float64Array(N);
    const edges = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, Infinity];
    const hist = edges.slice(0, -1).map((lo, i) => ({ lo, hi: edges[i + 1], byType: [0, 0, 0, 0] }));
    const saleYears = new Float64Array(Math.ceil(F.horizon)), exitTimes = [];
    let worse = 0, belowCost = 0, sellSoon = 0;
    for (let p = 0; p < N; p++) {
      mix[sim.type[p]]++; pol[how[p]]++;
      mult[p] = sim.gross[p] / Math.max(fvOpm, 1);
      const bin = edges.findIndex((e, i) => mult[p] >= e && mult[p] < edges[i + 1]);
      hist[bin].byType[sim.type[p]] += 1 / N;
      if (cfHold[p] < S0net) worse++;
      if (sim.gross[p] < H.costBasis) belowCost++;
      if (sim.type[p] >= 1 && sim.type[p] <= 3) exitTimes.push(sim.event[p]);
      if (how[p] === 4) { const t = when[p] - M.lag; saleYears[Math.min(saleYears.length - 1, Math.floor(t - 1e-9))] += 1 / N; if (t <= 3) sellSoon++; }   // dated by the decision, not the cash
    }
    mult.sort();
    exitTimes.sort((a, b) => a - b);
    sellSoon /= N;

    let verdict;
    if (M.pClose < 0.05) verdict = 'Build liquidity';
    else if (fStar >= 0.95) verdict = 'Sell all';
    else if (fStar >= 0.1) verdict = 'Sell ' + Math.round(fStar * 100) + '%';
    else if (sr < 1 && sellSoon >= 0.35) verdict = 'Wait, then sell';
    else verdict = 'Hold';

    if (!hasBid) warnings.push('No live bid entered: the sale price assumes a ' + Math.round(M.noBidDiscount * 100) + '% discount to fair value.');
    const costs = feeOf(S0gross) + M.upfront;
    if (S0gross > 0 && costs > 0.15 * S0gross) warnings.push('Fees and sale costs take ' + Math.round(100 * costs / S0gross) + '% of the price. Fixed costs weigh less when this stake is sold together with others.');
    if (!(I.C.revenue > 0)) warnings.push('No revenue entered: the company is simulated as a single value process with ' + Math.round(I.C.vol * 100) + '% volatility.');
    if (mk.ageY > 2 && I.C.revenue > 0) warnings.push('The last round is ' + mk.ageY.toFixed(1) + ' years old, so today\'s value leans ' + Math.round((1 - mk.wRound) * 100) + '% on the revenue multiple.');
    if (se > 0.03) warnings.push('Simulation noise is ' + (se * 100).toFixed(1) + '% of the hold value. Raise the path count for a tighter answer.');
    if (fvOpm > 0.25 * F.wealth) warnings.push('This stake is over a quarter of the family wealth figure. Check that figure in Family settings.');

    const top = Math.max(3 * mk.e0, 1.5 * (mk.post || 0), 1), wf = [];
    for (let i = 0; i <= 60; i++) { const E = top * i / 60; wf.push([E, settle(ct0, E, false), settle(ct0, E, true)]); }
    const oursRank = Math.min(...I.cap.filter((r) => r.ours > 0).map((r) => (r.common ? Infinity : r.rank)));
    let prefAhead = 0, prefTotal = 0;
    for (let i = 0; i < ct0.n; i++) { prefTotal += ct0.L[i]; if (!ct0.common[i] && ct0.rank[i] < oursRank) prefAhead += ct0.L[i]; }

    return {
      asOf: new Date().toISOString().slice(0, 10), paths: N, bundles: sim.B, horizon: F.horizon,
      mark: { post: mk.post, eOpm: mk.eOpm, eRoll: mk.eRoll, eMult: mk.eMult, wRound: mk.wRound, e0: mk.e0, ageY: mk.ageY, lastClass: mk.lastClass, pLast: mk.pLast },
      process: proc,
      position: { shares: sh, ownership: ct0.oursConv / ct0.fd, liq0, fvOpm, prefAhead, prefTotal, costBasis: H.costBasis, heldYears: H.holdYears,
        classes: I.cap.map((r, i) => ({ name: r.name, payout: perClass[i], pref: ct0.L[i], ours: r.ours, rank: r.common ? null : r.rank })) },
      sale: { hasBid, bid: hasBid ? M.bid : S0gross / sh, S0gross, S0net, fee, pClose: M.pClose, discount: 1 - S0gross / Math.max(fvOpm, 1),
        costs, months: M.lag * 12, preTax: (S0gross - feeOf(S0gross)) * DF(M.lag) - tries },
      hold: { holdPV, vWait, se, irr, pWorse: worse / N, pBelowCost: belowCost / N, sellSoon,
        mix: mix.slice(0, 4).map((x) => x / N), policy: pol.map((x) => x / N),
        pct: { p10: quantile(mult, 0.1), p25: quantile(mult, 0.25), p50: quantile(mult, 0.5), p75: quantile(mult, 0.75), p90: quantile(mult, 0.9), mean: mean(mult) },
        hist, saleYears: Array.from(saleYears), medianExit: quantile(exitTimes, 0.5) },
      decision: { verdict, sr, srBand: [vHi > 0 ? S0net / vHi : sr, sr, vLo > 0 ? S0net / vLo : sr], agree,
        fStar, fHedge, fUtil, driver: fStar === 0 ? 'none' : fUtil > fHedge ? 'concentration' : 'price',
        fs: FS, ceGain, gainUtil: ceGain[jStar], ladder, vLo, vHi, hurdle: F.hurdle },
      waterfall: wf,
      warnings,
    };
  }

  // What moves the answer, and where each assumption comes from: rerun with one input pushed lower and higher.
  // src says whether the value was entered ('input'), is a family setting ('family') or is the engine's placeholder ('default').
  function tornado(raw, onProgress) {
    const I = normalize(raw), lite = { paths: 4000, bundles: 8 };
    const clone = () => JSON.parse(JSON.stringify(raw));
    const set = (path, v) => (r) => { const [a, b] = path.split('.'); r[a] = r[a] || {}; r[a][b] = v; return r; };
    const bids = (k) => (r) => { r.market = r.market || {}; r.market.bids = (r.market.bids || []).map((b) => Object.assign({}, b, { price: num(b.price, 0) * k })); return r; };
    const given = (path) => {
      const [a, b] = path.split('.'), v = raw && raw[a] ? raw[a][b] : undefined;
      if (a === 'family') return 'family';
      return (v !== undefined && v !== null && v !== '') || (path === 'market.bidSpread' && I.M.spreadFromBids) ? 'input' : 'default';
    };
    const C = I.C, M = I.M, F = I.F, rev = C.revenue > 0;
    const L = (label, path, kind, value, lo, hi, span) => ({ label, path, kind, value, lo, hi, span });
    const vars = [
      rev && L('Revenue growth', 'company.growth', 'pct', C.growth, C.growth - 0.1, C.growth + 0.1, '±10 pts'),
      rev && L('Cash-flow margin', 'company.fcfMargin', 'pct', C.fcfMargin, C.fcfMargin - 0.1, C.fcfMargin + 0.1, '±10 pts'),
      rev && L('Sector multiple', 'company.compMultiple', 'x', C.compMultiple, C.compMultiple * 0.7, C.compMultiple * 1.3, '±30%'),
      rev && L('Private-company discount', 'company.privateDiscount', 'pct', C.privateDiscount, Math.max(0, C.privateDiscount - 0.1), C.privateDiscount + 0.1, '±10 pts'),
      rev && L('Long-run growth', 'company.longRunGrowth', 'pct', C.longRunGrowth, C.longRunGrowth - 0.03, C.longRunGrowth + 0.03, '±3 pts'),
      rev && L('Mature cash-flow margin', 'company.targetMargin', 'pct', C.targetMargin, C.targetMargin - 0.1, C.targetMargin + 0.1, '±10 pts'),
      rev && L('Revenue needed to IPO', 'company.ipoRevenue', 'money', C.ipoRevenue, C.ipoRevenue * 0.5, C.ipoRevenue * 2, '×0.5 / ×2'),
      !rev && L('Value growth', 'company.valueGrowth', 'pct', C.valueGrowth, C.valueGrowth - 0.1, C.valueGrowth + 0.1, '±10 pts'),
      !rev && L('Volatility', 'company.vol', 'pct', C.vol, C.vol * 0.7, C.vol * 1.3, '±30%'),
      L('Acquisition likelihood', 'company.maHazard', 'pct', C.maHazard, C.maHazard * 0.5, C.maHazard * 2, '×0.5 / ×2'),
      L('IPO likelihood', 'company.ipoHazard', 'pct', C.ipoHazard, C.ipoHazard * 0.5, C.ipoHazard * 2, '×0.5 / ×2'),
      L('Disruption shock', 'company.disruption', 'pct', C.disruption, C.disruption / 3, C.disruption * 3, '×⅓ / ×3'),
      M.bid > 0 && L('Best bid', 'market.bids', 'perShare', M.bid, bids(0.85), bids(1.15), '±15%'),
      L('Typical secondary discount', 'market.noBidDiscount', 'pct', M.noBidDiscount, Math.max(0, M.noBidDiscount - 0.15), M.noBidDiscount + 0.15, '±15 pts'),
      L('Serious buyers', 'market.buyers', 'num', M.buyers, M.buyers / 2, M.buyers * 2, '×0.5 / ×2'),
      L('Spread of bids', 'market.bidSpread', 'pct', M.spread, M.spread / 2, M.spread * 2, '×0.5 / ×2'),
      L('Exit window', 'family.window', 'window', F.window, -1, 1, 'shut / open'),
      L('Hurdle rate', 'family.hurdle', 'pct', F.hurdle, Math.max(0, F.hurdle - 0.05), F.hurdle + 0.05, '±5 pts'),
      L('Risk aversion', 'family.gamma', 'num', F.gamma, 1, 6, 'γ 1 / 6'),
    ].filter(Boolean);
    const pick = (res) => ({ sr: res.decision.sr, f: res.decision.fStar, pSale: res.process.pSale });
    const patch = (path, x) => (typeof x === 'function' ? x : set(path, x));
    const base = pick(run(raw, lite)), rows = [];
    vars.forEach((v, i) => {
      rows.push({ label: v.label, span: v.span, kind: v.kind, value: v.value, src: given(v.path),
        lo: pick(run(patch(v.path, v.lo)(clone()), lite)), hi: pick(run(patch(v.path, v.hi)(clone()), lite)) });
      if (onProgress) onProgress(i + 1, vars.length);
    });
    return { base, rows };
  }

  const api = { run, tornado, normalize, compileCap, settle, distribute, addRound, opmValue, backsolve, invNorm, normCdf, makeTax };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof importScripts === 'function') {
    self.onmessage = (e) => {
      const { id, kind, input } = e.data;
      try {
        const result = kind === 'tornado' ? tornado(input, (k, n) => self.postMessage({ id, progress: k / n })) : run(input);
        self.postMessage({ id, result });
      } catch (err) {
        self.postMessage({ id, error: String((err && err.message) || err) });
      }
    };
  } else if (typeof window !== 'undefined') {
    window.ExitEngine = api;   // main-thread fallback when workers are unavailable
  }
})();
