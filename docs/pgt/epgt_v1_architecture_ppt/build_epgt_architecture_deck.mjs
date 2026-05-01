import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  layers,
  panel,
  text,
  shape,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
  auto,
} = await import(
  "file:///C:/Users/isayi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs"
);

const ROOT = "F:/HUAWEI_Theise/Thesis Transformer version1/docs/pgt/epgt_v1_architecture_ppt";
const OUT = path.join(ROOT, "output");
const PREVIEW = path.join(ROOT, "previews");
const REPORTS = path.join(ROOT, "reports");
await mkdir(OUT, { recursive: true });
await mkdir(PREVIEW, { recursive: true });
await mkdir(REPORTS, { recursive: true });

const W = 1920;
const H = 1080;
const C = {
  ink: "#102033",
  muted: "#53657A",
  faint: "#7890A8",
  bg: "#F6F8FB",
  white: "#FFFFFF",
  line: "#D8E0EA",
  blue: "#2A6FDB",
  blue2: "#D8E8FF",
  teal: "#159A9C",
  teal2: "#D9F2F2",
  green: "#2E9D57",
  green2: "#DCF4E4",
  orange: "#D87922",
  orange2: "#FBE7D2",
  purple: "#6B55C8",
  purple2: "#E8E3FF",
  red: "#C44E52",
  dark: "#12253A",
};

const FONT = "Microsoft YaHei";
const MONO = "Consolas";

const presentation = Presentation.create({
  slideSize: { width: W, height: H },
});

function compose(slide, child) {
  slide.compose(child, {
    frame: { left: 0, top: 0, width: W, height: H },
    baseUnit: 8,
  });
}

function bg(children, name = "slide-root") {
  return layers({ name, width: fill, height: fill }, [
    shape({
      name: `${name}-bg`,
      width: fill,
      height: fill,
      fill: C.bg,
      line: { fill: C.bg, width: 0 },
    }),
    ...children,
  ]);
}

function titleBlock(title, subtitle, extra = {}) {
  return column(
    {
      name: extra.name ?? "title-block",
      width: fill,
      height: hug,
      gap: 12,
    },
    [
      text(title, {
        name: "slide-title",
        width: fill,
        height: hug,
        style: {
          fontFace: FONT,
          fontSize: extra.titleSize ?? 46,
          bold: true,
          color: extra.color ?? C.ink,
        },
      }),
      subtitle
        ? text(subtitle, {
            name: "slide-subtitle",
            width: wrap(extra.subtitleWidth ?? 1320),
            height: hug,
            style: {
              fontFace: FONT,
              fontSize: extra.subtitleSize ?? 23,
              color: extra.subColor ?? C.muted,
              lineSpacing: 1.15,
            },
          })
        : rule({ name: "title-rule", width: fixed(180), stroke: C.blue, weight: 5 }),
    ].filter(Boolean),
  );
}

function chip(label, color = C.blue, fillColor = C.blue2, width = 160) {
  return panel(
    {
      name: `chip-${label}`,
      width: fixed(width),
      height: fixed(38),
      padding: { x: 12, y: 5 },
      fill: fillColor,
      line: { fill: color, width: 1 },
      borderRadius: "rounded-full",
      align: "center",
      justify: "center",
    },
    text(label, {
      name: `chip-text-${label}`,
      width: fill,
      height: hug,
      style: { fontFace: FONT, fontSize: 15, bold: true, color },
    }),
  );
}

function nodeBox(name, title, lines, opts = {}) {
  return panel(
    {
      name,
      width: opts.width ?? fill,
      height: opts.height ?? hug,
      padding: opts.padding ?? { x: 22, y: 18 },
      fill: opts.fill ?? C.white,
      line: { fill: opts.line ?? C.line, width: opts.lineWidth ?? 1.2 },
      borderRadius: "rounded-lg",
      shadow: opts.shadow,
    },
    column({ width: fill, height: hug, gap: 8 }, [
      text(title, {
        name: `${name}-title`,
        width: fill,
        height: hug,
        style: {
          fontFace: FONT,
          fontSize: opts.titleSize ?? 23,
          bold: true,
          color: opts.titleColor ?? C.ink,
        },
      }),
      ...lines.map((line, idx) =>
        text(line, {
          name: `${name}-line-${idx}`,
          width: fill,
          height: hug,
          style: {
            fontFace: line.includes("[") || line.includes("=") ? MONO : FONT,
            fontSize: opts.lineSize ?? 17,
            color: opts.textColor ?? C.muted,
            lineSpacing: 1.08,
          },
        }),
      ),
    ]),
  );
}

function arrow(label = "→", color = C.faint) {
  return text(label, {
    name: `arrow-${label}`,
    width: fixed(52),
    height: hug,
    style: { fontFace: FONT, fontSize: 34, bold: true, color },
  });
}

function formulaBlock(name, formulaLines, opts = {}) {
  return panel(
    {
      name,
      width: opts.width ?? fill,
      height: hug,
      padding: { x: 26, y: 22 },
      fill: opts.fill ?? "#0F2437",
      line: { fill: opts.line ?? "#0F2437", width: 0 },
      borderRadius: "rounded-lg",
    },
    column({ width: fill, height: hug, gap: 10 }, [
      ...formulaLines.map((line, idx) =>
        text(line, {
          name: `${name}-${idx}`,
          width: fill,
          height: hug,
          style: {
            fontFace: MONO,
            fontSize: opts.fontSize ?? 22,
            color: opts.color ?? "#FFFFFF",
          },
        }),
      ),
    ]),
  );
}

function footer(slideNum) {
  return row(
    { name: "footer", width: fill, height: hug, align: "center" },
    [
      text("EPGT-v1 architecture | physics-guided Transformer", {
        name: "footer-left",
        width: fill,
        height: hug,
        style: { fontFace: FONT, fontSize: 12, color: C.faint },
      }),
      text(String(slideNum).padStart(2, "0"), {
        name: "footer-page",
        width: fixed(42),
        height: hug,
        style: { fontFace: MONO, fontSize: 13, color: C.faint },
      }),
    ],
  );
}

function addCover() {
  const slide = presentation.slides.add();
  compose(
    slide,
    panel(
      {
        name: "cover-root",
        width: fill,
        height: fill,
        padding: { x: 120, y: 104 },
        fill: "#F4F8FC",
        line: { fill: "#F4F8FC", width: 0 },
        borderRadius: "rounded-none",
      },
      column(
        {
          name: "cover-content",
          width: fill,
          height: hug,
          gap: 22,
        },
      [
          row({ name: "cover-chip-row", width: fill, height: hug, gap: 14 }, [
            chip("MIMO-OFDM", C.teal, C.teal2, 168),
            chip("Physics-Guided Attention", C.blue, C.blue2, 300),
          ]),
          text("EPGT-v1", {
            name: "cover-title",
            width: fill,
            height: hug,
            style: { fontFace: MONO, fontSize: 86, bold: true, color: C.ink },
          }),
          text("Physics-Guided Transformer\n网络架构与张量维度说明", {
            name: "cover-subtitle",
            width: wrap(1220),
            height: hug,
            style: { fontFace: FONT, fontSize: 38, bold: true, color: C.ink, lineSpacing: 1.05 },
          }),
          rule({ name: "cover-rule", width: fixed(320), stroke: C.teal, weight: 6 }),
          text("从稀疏观测 token 预测 effective-path 物理结构，再用 Gamma(q,i) 引导 full-grid query 读取最相关的观测点。", {
            name: "cover-promise",
            width: wrap(1180),
            height: hug,
            style: { fontFace: FONT, fontSize: 27, color: C.muted, lineSpacing: 1.18 },
          }),
        ],
      ),
    ),
  );
}

function addDimensions() {
  const slide = presentation.slides.add();
  const dims = [
    ["B", "batch size", "训练/评估批量"],
    ["N", "time points", "OFDM symbol 数"],
    ["K", "subcarriers", "载频 / 子载波数"],
    ["Nr", "RX antennas", "接收天线数"],
    ["Ns", "TX antennas", "发送天线数"],
    ["Leff", "effective paths", "估计器使用的有效路径数"],
    ["M", "observed tokens", "当前 M = 2K"],
    ["Q", "full-grid queries", "Q = NK"],
    ["D", "embedding dim", "Transformer 隐空间维度"],
  ];
  compose(
    slide,
    bg([
      column(
        { name: "s2-content", width: fill, height: fill, padding: { x: 82, y: 66 }, gap: 32 },
        [
          titleBlock("维度字典：物理维度 + 网络维度", "EPGT 的图里必须同时标清信道张量维度、token 数量，以及 cross-attention 的 query/context 维度。"),
          grid(
            {
              name: "dimension-grid",
              width: fill,
              height: grow(1),
              columns: [fr(1), fr(1), fr(1)],
              rows: [auto, auto, auto],
              columnGap: 22,
              rowGap: 18,
            },
            dims.map(([sym, en, cn], idx) =>
              panel(
                {
                  name: `dim-card-${sym}`,
                  width: fill,
                  height: fixed(132),
                  padding: { x: 22, y: 16 },
                  fill: idx < 6 ? C.white : "#F9FBFF",
                  line: { fill: idx < 6 ? C.line : C.blue2, width: 1.2 },
                  borderRadius: "rounded-lg",
                },
                row({ width: fill, height: fill, gap: 18, align: "center" }, [
                  text(sym, {
                    name: `dim-symbol-${sym}`,
                    width: fixed(104),
                    height: hug,
                    style: { fontFace: MONO, fontSize: 36, bold: true, color: idx < 6 ? C.blue : C.teal },
                  }),
                  column({ width: fill, height: hug, gap: 8 }, [
                    text(en, { name: `dim-en-${sym}`, width: fill, height: hug, style: { fontFace: MONO, fontSize: 16, color: C.faint } }),
                    text(cn, { name: `dim-cn-${sym}`, width: fill, height: hug, style: { fontFace: FONT, fontSize: 20, bold: true, color: C.ink } }),
                  ]),
                ]),
              ),
            ),
          ),
          formulaBlock(
            "shape-summary",
            [
              "X_tok: [B, M, F_in],   F_in = 2Nr + 2Ns + 2 + 1 + 1",
              "Gamma: [B, Q, M],      Q = N*K",
              "H_hat: [B, Nr, Ns, N, K]",
            ],
            { fill: "#FFFFFF", line: C.line, color: C.ink, fontSize: 22 },
          ),
          footer(2),
        ],
      ),
    ]),
  );
}

function addPipeline() {
  const slide = presentation.slides.add();
  compose(
    slide,
    bg([
      column(
        { name: "s3-content", width: fill, height: fill, padding: { x: 72, y: 60 }, gap: 28 },
        [
          titleBlock("总体流程：先估物理结构，再用物理结构引导 attention", "一页里只保留主干路径：observed tokens → encoder → heads → Gamma → cross-attention → LS/reconstruction。"),
          row(
            { name: "pipeline-row-1", width: fill, height: hug, gap: 12, align: "center" },
            [
              nodeBox("p-observed", "Sparse observations", ["Y_obs: [B,Nr,2,K]", "X_obs: [B,Ns,2,K]"], { width: fixed(270), titleColor: C.blue }),
              arrow(),
              nodeBox("p-token", "Observation tokens", ["X_tok: [B,M,F_in]", "M = 2K"], { width: fixed(275), titleColor: C.blue }),
              arrow(),
              nodeBox("p-encoder", "ObservationEncoder", ["global token + tokens", "Z_obs: [B,M,D]", "z_g: [B,D]"], { width: fixed(315), titleColor: C.teal }),
              arrow(),
              nodeBox("p-heads", "Path/global heads", ["alpha,d,nu: [B,Leff]", "cfo,epsilon"], { width: fixed(300), titleColor: C.purple }),
              arrow(),
              nodeBox("p-gamma", "Physics bias", ["Gamma(q,i): [B,Q,M]", "Q = NK"], { width: fixed(290), titleColor: C.orange }),
            ],
          ),
          row(
            { name: "pipeline-row-2", width: fill, height: hug, gap: 12, align: "center" },
            [
              panel(
                { name: "query-lane", width: fixed(410), height: fixed(152), padding: { x: 22, y: 16 }, fill: C.blue2, line: { fill: "#B9D3F5", width: 1.2 }, borderRadius: "rounded-lg" },
                column({ width: fill, height: hug, gap: 8 }, [
                  text("Full-grid query lane", { name: "query-title", width: fill, height: hug, style: { fontFace: FONT, fontSize: 23, bold: true, color: C.blue } }),
                  text("C_q: [Q,2],  Q = N*K\nZ_q: [B,Q,D]", { name: "query-lines", width: fill, height: hug, style: { fontFace: MONO, fontSize: 19, color: C.ink } }),
                ]),
              ),
              arrow("＋", C.blue),
              nodeBox("p-cross", "Physics-guided cross-attention", ["query: [B,Q,D]", "context: [B,M,D]", "bias: [B,Q,M]"], { width: fixed(405), titleColor: C.teal, fill: "#F7FFFE", line: "#BBDDDD" }),
              arrow(),
              nodeBox("p-ls", "Complex LS recovery", ["G_hat: [B,Leff,Nr,Ns]"], { width: fixed(315), titleColor: C.green, fill: "#F7FFF9", line: "#B8DEC5" }),
              arrow(),
              nodeBox("p-hhat", "Full-grid channel", ["H_hat: [B,Nr,Ns,N,K]"], { width: fixed(360), titleColor: C.red, fill: "#FFF8F8", line: "#E9C6C8" }),
            ],
          ),
          formulaBlock(
            "pipeline-thesis",
            ["EPGT is not only a deeper Transformer:", "predicted effective paths -> Gamma(q,i) -> guided reading of observed REs"],
            { fill: "#12253A", fontSize: 24 },
          ),
          footer(3),
        ],
      ),
    ]),
  );
}

function addHeads() {
  const slide = presentation.slides.add();
  compose(
    slide,
    bg([
      column(
        { name: "s4-content", width: fill, height: fill, padding: { x: 80, y: 64 }, gap: 28 },
        [
          titleBlock("参数头：global token 是全局物理状态容器", "Baseline 只做 mean pooling；EPGT 用 learnable global token 承载 CFO、采样偏移和 effective-path 参数。"),
          grid(
            { name: "heads-grid", width: fill, height: grow(1), columns: [fr(0.9), fr(1.05), fr(1.05)], columnGap: 26 },
            [
              nodeBox("heads-input", "Encoder output", ["z_g: [B,D]", "Z_obs: [B,M,D]"], { height: fill, titleColor: C.teal, fill: "#F7FFFE", line: "#BBDDDD", titleSize: 26, lineSize: 21 }),
              panel(
                { name: "global-head", width: fill, height: fill, padding: { x: 28, y: 26 }, fill: C.white, line: { fill: C.line, width: 1.2 }, borderRadius: "rounded-lg" },
                column({ width: fill, height: hug, gap: 18 }, [
                  chip("Global head", C.blue, C.blue2, 190),
                  text("输出公共物理量", { name: "global-head-title", width: fill, height: hug, style: { fontFace: FONT, fontSize: 30, bold: true, color: C.ink } }),
                  formulaBlock("global-head-formula", ["tau0:    [B]", "cfo:     [B]", "epsilon: [B,Nr]"], { fill: "#F4F8FC", line: C.line, color: C.ink, fontSize: 25 }),
                  text("v1 规范：tau0 固定 common_delay；epsilon[:,0] = 0，只学习相对接收天线采样偏移。", {
                    name: "global-head-note",
                    width: fill,
                    height: hug,
                    style: { fontFace: FONT, fontSize: 21, color: C.muted, lineSpacing: 1.18 },
                  }),
                ]),
              ),
              panel(
                { name: "path-head", width: fill, height: fill, padding: { x: 28, y: 26 }, fill: C.white, line: { fill: C.line, width: 1.2 }, borderRadius: "rounded-lg" },
                column({ width: fill, height: hug, gap: 18 }, [
                  chip("Effective-path head", C.purple, C.purple2, 250),
                  text("输出 delay-Doppler 原子", { name: "path-head-title", width: fill, height: hug, style: { fontFace: FONT, fontSize: 30, bold: true, color: C.ink } }),
                  formulaBlock("path-head-formula", ["alpha: [B,Leff]", "d:     [B,Leff]", "nu:    [B,Leff]"], { fill: "#F7F5FF", line: C.purple2, color: C.ink, fontSize: 25 }),
                  text("v1 规范：alpha 是 path gate；nu 做加权中心化，避免和 CFO 发生加法歧义。", {
                    name: "path-head-note",
                    width: fill,
                    height: hug,
                    style: { fontFace: FONT, fontSize: 21, color: C.muted, lineSpacing: 1.18 },
                  }),
                ]),
              ),
            ],
          ),
          footer(4),
        ],
      ),
    ]),
  );
}

function addGamma() {
  const slide = presentation.slides.add();
  compose(
    slide,
    bg([
      column(
        { name: "s5-content", width: fill, height: fill, padding: { x: 76, y: 60 }, gap: 24 },
        [
          titleBlock("核心：Gamma(q,i) 把 delay-Doppler 相干性注入 cross-attention", "对每个待估位置 q，EPGT 会判断哪些观测点 i 在当前 effective-path 结构下更相干。"),
          grid(
            { name: "gamma-grid", width: fill, height: grow(1), columns: [fr(1.03), fr(0.97)], columnGap: 28 },
            [
              column({ name: "gamma-left", width: fill, height: fill, gap: 18 }, [
                formulaBlock(
                  "gamma-kernel",
                  [
                    "K(q,i) = | sum_l alpha_l",
                    "    exp(j 2pi nu_l Delta_t_qi)",
                    "    exp(-j 2pi d_l Delta_f_qi) |",
                    "",
                    "Gamma(q,i) = lambda * log(eps + K(q,i))",
                  ],
                  { fill: "#0F2437", fontSize: 24 },
                ),
                formulaBlock(
                  "gamma-score",
                  ["score(q,i) = Q_q K_i^T / sqrt(d) + Gamma(q,i)", "Attn(q,obs) = softmax(score) V_obs"],
                  { fill: "#FFFFFF", line: C.line, color: C.ink, fontSize: 23 },
                ),
              ]),
              column({ name: "gamma-right", width: fill, height: fill, gap: 16 }, [
                row({ width: fill, height: hug, gap: 12, align: "center" }, [
                  chip("q = (nq,kq)", C.blue, C.blue2, 190),
                  arrow("↔", C.faint),
                  chip("i = (ni,ki)", C.orange, C.orange2, 190),
                ]),
                nodeBox("delta-box", "相对时频差", ["Delta_t_qi = t_nq - t_ni", "Delta_f_qi = f_kq - f_ki"], { fill: "#FFFFFF", titleColor: C.orange, lineSize: 22 }),
                nodeBox("gamma-shapes", "关键张量维度", ["C_q:     [Q,2],   Q = N*K", "C_obs:   [B,M,2]", "alpha,d,nu: [B,Leff]", "Gamma:   [B,Q,M]"], { fill: "#FFFFFF", titleColor: C.teal, lineSize: 22 }),
                panel(
                  { name: "gamma-meaning", width: fill, height: hug, padding: { x: 24, y: 20 }, fill: C.teal2, line: { fill: "#BBDDDD", width: 1.2 }, borderRadius: "rounded-lg" },
                  text("直观含义：如果观测点 i 和 query q 能由同一组 delay-Doppler path 解释，Gamma 就提高 q 对 i 的 attention logit。", {
                    name: "gamma-meaning-text",
                    width: fill,
                    height: hug,
                    style: { fontFace: FONT, fontSize: 24, bold: true, color: C.ink, lineSpacing: 1.18 },
                  }),
                ),
              ]),
            ],
          ),
          footer(5),
        ],
      ),
    ]),
  );
}

function addReconstruction() {
  const slide = presentation.slides.add();
  compose(
    slide,
    bg([
      column(
        { name: "s6-content", width: fill, height: fill, padding: { x: 76, y: 60 }, gap: 24 },
        [
          titleBlock("后处理：用预测的非线性参数做 LS，再显式重构 H", "EPGT 的输出不是黑盒 H，而是 nonlinear physical parameters + LS 求出的 path gains。"),
          row(
            { name: "recon-row", width: fill, height: hug, gap: 14, align: "center" },
            [
              nodeBox("r-param", "Predicted params", ["cfo, epsilon", "alpha,d,nu"], { width: fixed(290), titleColor: C.purple }),
              arrow(),
              nodeBox("r-design", "Physics design matrix", ["Phi(d,nu,cfo,epsilon)", "from observed REs"], { width: fixed(360), titleColor: C.orange }),
              arrow(),
              nodeBox("r-ls", "Complex LS", ["G_hat: [B,Leff,Nr,Ns]"], { width: fixed(330), titleColor: C.green }),
              arrow(),
              nodeBox("r-h", "Reconstruction", ["H_hat: [B,Nr,Ns,N,K]"], { width: fixed(390), titleColor: C.red }),
            ],
          ),
          formulaBlock(
            "h-formula",
            [
              "H[r,t,n,k] = exp(-j2pi f_k(tau0 + epsilon_r)) * exp(j2pi cfo t_n)",
              "           * sum_l G[l,r,t] exp(j2pi nu_l t_n) exp(-j2pi f_k d_l)",
            ],
            { fill: "#12253A", fontSize: 22 },
          ),
          grid(
            { name: "compare-grid", width: fill, height: grow(1), columns: [fr(1), fr(1)], columnGap: 28 },
            [
              panel(
                { name: "baseline-panel", width: fill, height: fill, padding: { x: 28, y: 24 }, fill: "#FFFFFF", line: { fill: C.line, width: 1.2 }, borderRadius: "rounded-lg" },
                column({ width: fill, height: hug, gap: 18 }, [
                  chip("Baseline HybridTransformer", C.faint, "#EEF2F6", 300),
                  text("observed-token self-attention\n→ mean pooling\n→ parameter head\n→ LS reconstruction", {
                    name: "baseline-flow",
                    width: fill,
                    height: hug,
                    style: { fontFace: MONO, fontSize: 25, color: C.ink, lineSpacing: 1.32 },
                  }),
                  text("没有 full-grid query，也没有由 delay-Doppler 结构产生的 attention bias。", {
                    name: "baseline-note",
                    width: fill,
                    height: hug,
                    style: { fontFace: FONT, fontSize: 22, color: C.muted, lineSpacing: 1.18 },
                  }),
                ]),
              ),
              panel(
                { name: "epgt-panel", width: fill, height: fill, padding: { x: 28, y: 24 }, fill: "#F7FFFE", line: { fill: "#BBDDDD", width: 1.2 }, borderRadius: "rounded-lg" },
                column({ width: fill, height: hug, gap: 18 }, [
                  chip("EPGT-v1", C.teal, C.teal2, 160),
                  text("path/global heads\n→ Gamma(q,i)\n→ guided cross-attention\n→ refined physical state\n→ LS reconstruction", {
                    name: "epgt-flow",
                    width: fill,
                    height: hug,
                    style: { fontFace: MONO, fontSize: 25, color: C.ink, lineSpacing: 1.32 },
                  }),
                  text("性能提升来自更强的归纳偏置：query 不是平均读所有观测，而是优先读物理相干的观测。", {
                    name: "epgt-note",
                    width: fill,
                    height: hug,
                    style: { fontFace: FONT, fontSize: 22, color: C.muted, lineSpacing: 1.18 },
                  }),
                ]),
              ),
            ],
          ),
          footer(6),
        ],
      ),
    ]),
  );
}

addCover();
addDimensions();
addPipeline();
addHeads();
addGamma();
addReconstruction();

const pptxPath = path.join(OUT, "epgt_v1_architecture.pptx");
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(pptxPath);

  const previewPaths = [];
for (let i = 0; i < presentation.slides.count; i += 1) {
  const slide = presentation.slides.getItem(i);
  const png = await slide.export({ format: "png" });
  const pngPath = path.join(PREVIEW, `slide_${String(i + 1).padStart(2, "0")}.png`);
  await writeFile(pngPath, Buffer.from(await png.arrayBuffer()));
  previewPaths.push(pngPath);

  const layout = await slide.export({ format: "layout" });
  await writeFile(path.join(REPORTS, `slide_${String(i + 1).padStart(2, "0")}.layout.json`), JSON.stringify(layout, null, 2), "utf8");
}

await writeFile(
  path.join(REPORTS, "build_summary.json"),
  JSON.stringify({ pptxPath, previewPaths, slideCount: presentation.slides.length }, null, 2),
  "utf8",
);

console.log(JSON.stringify({ pptxPath, previewPaths, slideCount: presentation.slides.length }, null, 2));
