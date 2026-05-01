import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const { PresentationFile, FileBlob } = await import(
  "file:///C:/Users/isayi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs"
);

const ROOT = "F:/HUAWEI_Theise/Thesis Transformer version1/docs/pgt/epgt_v1_architecture_ppt";
const pptxPath = path.join(ROOT, "output", "epgt_v1_architecture.pptx");
const parityDir = path.join(ROOT, "parity");
await mkdir(parityDir, { recursive: true });

const pptxBlob = await FileBlob.load(pptxPath);
const presentation = await PresentationFile.importPptx(pptxBlob);
const previewPaths = [];

for (let i = 0; i < presentation.slides.count; i += 1) {
  const slide = presentation.slides.getItem(i);
  const png = await slide.export({ format: "png" });
  const pngPath = path.join(parityDir, `pptx_slide_${String(i + 1).padStart(2, "0")}.png`);
  await writeFile(pngPath, Buffer.from(await png.arrayBuffer()));
  previewPaths.push(pngPath);
}

console.log(JSON.stringify({ slideCount: presentation.slides.count, previewPaths }, null, 2));
