# Skill: Editable PowerPoint slides that match an HTML deck layout

Build a slide deck once, as HTML, and get BOTH a web deck and an **editable** `.pptx` whose every text box, image,
bullet and rule sits where it sits in the HTML — no screenshots pasted into PowerPoint, no layout re-done in
python-pptx. Use for any study deck made from model figures (LUTO runs, task-run comparisons, papers' figure decks),
when adding a slide, making a variant of an existing slide, or rebuilding a deck.

Reference implementation (copy it into a new project and adapt): `jinzhu_inspect_code/Explore_CMA_narrative/slides/` —
`make_slides.py` (artboards), `make_plain_deck.py` (HTML deck), `make_pptx.py` + `measure.mjs` (pptx).

## 1. The idea

* **One artboard per slide**: a fixed 1280 × 720 px HTML page (`<name>.dc.html`), laid out with flexbox/grid from a few
  Python helpers — `frame(inner, pad, gap)`, `header(tag, title, size)`, `bullets(items, size)`, `note(text)`,
  `img(src, h=...)`, `slide(tag, title, figure_html, items, foot)` (figure left, bullets right), `divider(...)`,
  `two_col(...)`. A dict `S[name] = frame(...)` holds every slide; `ORDER` is the deck; `canvas.json` lists the boards.
* **HTML deck**: the artboard bodies stacked into one file, images inlined as data URIs.
* **pptx by measurement, not re-layout**: each artboard is opened in **headless Microsoft Edge** (driven over the
  DevTools protocol by `measure.mjs`; Edge's `--dump-dom` prints nothing on this machine), with the web fonts swapped for
  the fonts PowerPoint will render (Georgia for serif, Segoe UI for sans). A script in the page walks the DOM and returns
  every leaf box — text (runs with bold / italic, size, weight, colour, alignment, line-height, letter-spacing), images,
  small filled shapes (bullet dots), border rules — and python-pptx places a native shape at the same coordinates
  (1 CSS px = 9525 EMU, so 1280 × 720 px = a 13.333 × 7.5 in 16:9 slide). Tuning that keeps PowerPoint's wrap identical
  to the browser's: kerning on every run, exact line pitch in points, font size × 0.97, ~6 % box slack.
  SVG figures go in as vectors (svgBlip) with the PNG sibling as the fallback picture.

**Consequence: lay a slide out in the artboard HTML, never in python-pptx.** Change the HTML, rebuild, and the pptx follows.

## 2. Converting the deck to editable slides WITHOUT skew — read this before touching `make_pptx.py`

The first conversion of a deck usually comes out **skewed**: text boxes shifted or stretched, lines wrapping in
different places, bullets drifting away from their dots, titles spilling onto a second line, figures squashed. Each
symptom has one cause, and each cause has a fixed rule. Do **all** of them — dropping any one brings the skew back.

| # | symptom in the pptx | cause | rule (reference implementation) |
|---|---|---|---|
| 1 | everything stretched or squeezed sideways, figures wrong aspect | python-pptx's default slide is **4:3, 10 × 7.5 in**; the artboard is 16:9 | set `prs.slide_width = 1280 * 9525`, `prs.slide_height = 720 * 9525` EMU (13.333 × 7.5 in) **before** adding any slide; convert every px with the one factor `PX = 9525` EMU — never inches, never points for positions |
| 2 | every box shifted right / down by a few px | the measuring page has the browser's default `body { margin: 8px }`, or a viewport / device-scale other than 1280 × 720 @ 1× | measuring page: `html,body{margin:0;padding:0;overflow:hidden;width:1280px;height:720px}`; Edge window `--window-size=1280,720` **and** CDP `Emulation.setDeviceMetricsOverride({width:1280,height:720,deviceScaleFactor:1})` |
| 3 | lines wrap at different words; boxes too short so text overflows into the next box | boxes were measured with **web fonts** (Google fonts: Source Serif 4, IBM Plex Sans) that PowerPoint does not have; PowerPoint substitutes a font with other widths | measure with the fonts PowerPoint will render: strip the `fonts.googleapis` `<link>` and replace the web family names by Georgia / Segoe UI in the page **before measuring** (`page_for`); write the same font names into each run |
| 4 | still a few lines wrap one word early | PowerPoint sets Georgia / Segoe UI slightly wider than Chromium, and does not kern by default | each run: font size × `FONT_SCALE = 0.97`, `kern="100"` on every run's `rPr`; each text box ~6 % wider than measured (left-aligned grows right, right-aligned grows left; clipped at the slide edge) |
| 5 | text sits too low / too high in its box; multi-line blocks taller than in HTML | python-pptx text-frame defaults: insets 0.1 in left/right and 0.05 in top/bottom, auto-fit on, middle anchor, and PowerPoint's own line spacing | `margin_*` = the element's CSS padding (usually 0), `auto_size = None`, `word_wrap = True`, `vertical_anchor = TOP`, and `line_spacing = Pt(line-height_px × 0.75)` — the browser's exact line pitch in points (`normal` = 1.2 × font size) |
| 6 | font sizes too big by a third | CSS px treated as pt | font size in pt = px × 0.75 (96 dpi); same for line widths and letter-spacing (`spc` = px × 75, in 1/100 pt) |
| 7 | boxes measured as 0 × 0 or images at the wrong height; layout differs run to run | measured before the page finished: images not decoded, fonts not loaded | images embedded as data URIs in the measuring page; wait for `document.readyState === 'complete' && document.fonts.status === 'loaded'`, then a short pause, then measure (`measure.mjs`) |
| 8 | text missing, merged, or duplicated; bold words lost | walking the DOM by tag instead of by layout | a TEXT box = an element whose children are all inline (text, `b/strong/i/em/span/a`) with non-blank text; its runs keep bold / italic from tag or computed weight ≥ 600 / italic style; apply `text-transform: uppercase` to the run text yourself; recurse into block children only |
| 9 | bullet dots, rules or dashed boxes missing or misplaced | only text and images were exported | export small filled boxes (≤ 40 px, the bullet dots) as shapes (oval when radius ≥ half the side), `border-top` / `border-bottom` as connector lines at the border's centre line, dashed left borders as dashed outlines |
| 10 | figures blurred, or the slide fails to open | SVG put in as a plain picture (python-pptx cannot) | add the PNG sibling as the picture, attach the SVG as PowerPoint's `svgBlip` extension (`{96DAC541-…}`) — vector in PowerPoint 2016+, PNG fallback elsewhere |
| 11 | shapes off the slide, negative extents, corrupt file | an artboard whose content overflows 720 px | skip boxes with `y ≥ 720` or width ≤ 1 px; fix the overflow in the HTML (section 5 "Fit"), never in python-pptx |

### The rules as code (trimmed from the reference `make_pptx.py` / `measure.mjs`)

**Rule 1 — slide size and the one unit:**
```python
PX = 9525                                    # EMU per CSS px at 96 dpi
prs = Presentation()
prs.slide_width, prs.slide_height = Emu(1280 * PX), Emu(720 * PX)   # BEFORE add_slide; default is 4:3
slide = prs.slides.add_slide(prs.slide_layouts[6])                  # blank layout: no placeholders
```

**Rules 2 + 3 + 7 — the measuring page** (artboard body, images inlined, PowerPoint's fonts, no body margin):
```python
helmet = re.sub(r'<link[^>]*fonts\.googleapis[^>]*>', "", helmet)  # drop the web fonts ...
helmet = helmet.replace("Source Serif 4", "Georgia").replace("IBM Plex Sans", "Segoe UI")
body = body.replace("Source Serif 4", "Georgia").replace("IBM Plex Sans", "Segoe UI")   # ... inline styles too
body = re.sub(r'src="([^"]+)"', lambda m: f'src="{embed(m.group(1))}" data-src="{m.group(1)}"', body)  # data URI; keep the name
page = (f"<!doctype html><html><head><meta charset='utf-8'>{helmet}"
        "<style>html,body{margin:0;padding:0;overflow:hidden;width:1280px;height:720px}</style></head>"
        f"<body>{body}<script>{MEASURE_JS}</script></body></html>")
```
```js
// measure.mjs — fixed viewport at 1x, then wait for load AND fonts before measuring
await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false }, sessionId);
await cdp.send("Page.navigate", { url: pathToFileURL(page).href }, sessionId);
for (let i = 0; i < 100; i++) {
  const r = await cdp.send("Runtime.evaluate", { expression:
      "document.readyState === 'complete' && document.fonts.status === 'loaded'", returnByValue: true }, sessionId);
  if (r.result.value) break;
  await sleep(100);
}
await sleep(150);
const r = await cdp.send("Runtime.evaluate", { expression: "window.__measure()", returnByValue: true }, sessionId);
```

**Rule 8 — what counts as one text box** (inside the page's `window.__measure`):
```js
const inline = n => n.nodeType === 3 || (n.nodeType === 1 && ['B','STRONG','I','EM','SPAN','A'].includes(n.tagName));
const kids = Array.from(el.childNodes);
const isText = kids.some(n => n.nodeType === 3 && n.textContent.trim()) && kids.every(inline);
if (isText) {
  const r = el.getBoundingClientRect(), cs = getComputedStyle(el);
  out.texts.push({x: r.left, y: r.top, w: r.width, h: r.height, runs: runsOf(el, +cs.fontWeight >= 600, cs.fontStyle === 'italic'),
                  size: parseFloat(cs.fontSize), family: cs.fontFamily, color: rgb(cs.color), align: cs.textAlign,
                  lh: cs.lineHeight === 'normal' ? 1.2 * parseFloat(cs.fontSize) : parseFloat(cs.lineHeight),
                  pl: parseFloat(cs.paddingLeft), pr: parseFloat(cs.paddingRight),
                  pt: parseFloat(cs.paddingTop), pb: parseFloat(cs.paddingBottom)});
}
if (!isText) for (const k of el.children) walk(k);       // recurse into blocks only
```

**Rules 4 + 5 + 6 + 11 — placing a text box:**
```python
slack = t["w"] * 0.06                                    # PowerPoint sets type a touch wider than Chromium
x0 = max(0.0, t["x"] - (slack if t["align"] in ("right", "end") else 0))
w0 = min(t["w"] + slack, 1280 - x0)
if w0 <= 1 or t["y"] >= 720:                             # off the slide: skip, never a negative extent
    return
tb = slide.shapes.add_textbox(Emu(int(x0 * PX)), Emu(int(t["y"] * PX)), Emu(int(w0 * PX)), Emu(int(t["h"] * PX)))
tf = tb.text_frame
tf.word_wrap, tf.auto_size, tf.vertical_anchor = True, None, MSO_ANCHOR.TOP
tf.margin_left, tf.margin_right = Emu(int(t["pl"] * PX)), Emu(int(t["pr"] * PX))   # CSS padding, not 0.1 in
tf.margin_top, tf.margin_bottom = Emu(int(t["pt"] * PX)), Emu(int(t["pb"] * PX))
p = tf.paragraphs[0]
p.line_spacing = Pt(t["lh"] * 0.75)                      # the browser's exact line pitch, px -> pt
for r in t["runs"]:
    run = p.add_run(); run.text = r["t"]
    run.font.name = "Georgia" if "Georgia" in t["family"] else "Segoe UI"
    run.font.size = Pt(round(t["size"] * 0.75 * 0.97, 1))  # px -> pt, x FONT_SCALE
    run.font.bold, run.font.italic = bool(r["b"]), bool(r["i"])
    run._r.get_or_add_rPr().set("kern", "100")           # kern every size, as the browser does
```

**Rule 10 — SVG as a vector, PNG as the fallback:**
```python
pic = slide.shapes.add_picture(str(png), Emu(int(im["x"] * PX)), Emu(int(im["y"] * PX)),
                               width=Emu(int(im["w"] * PX)), height=Emu(int(im["h"] * PX)))
part = Part(PackURI("/ppt/media/svg1.svg"), "image/svg+xml", slide.part.package, svg.read_bytes())
rid = slide.part.relate_to(part, RT.IMAGE)
blip = pic._element.find(".//" + qn("a:blip"))
ext_lst = blip.find(qn("a:extLst"))
if ext_lst is None:
    ext_lst = etree.SubElement(blip, qn("a:extLst"))
ext = etree.SubElement(ext_lst, qn("a:ext"), uri="{96DAC541-7B7A-43D3-8B79-37D633B846F1}")
asvg = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
etree.SubElement(ext, f"{{{asvg}}}svgBlip", nsmap={"asvg": asvg}).set(qn("r:embed"), rid)
```
(One media part per SVG file, shared across slides — the reference keeps them in a dict.)

**The browser is Chromium.** Microsoft Edge is Chromium-based, so the layout engine doing the measuring is Chromium's
(Blink), driven over the **Chrome DevTools Protocol (CDP)**. `measure.mjs` is a ~60-line CDP client on node's built-in
`WebSocket`, with no Puppeteer / Playwright dependency. It uses Edge only because Edge ships with Windows, next to the
fonts PowerPoint uses. A Windows `chrome.exe` takes the same flags and should work as a drop-in (untested here). Two
rules: measure in a Chromium on the **same OS as PowerPoint**, so the fonts match; and never use Edge's `--dump-dom`
(it prints nothing on this machine). Run the page's own `window.__measure()` through CDP `Runtime.evaluate`.

**Measure in the browser, place in python-pptx, never re-lay-out.** No coordinate in the pptx is computed by python —
every `left / top / width / height` is a `getBoundingClientRect()` number × 9525. If a slide looks wrong in the pptx,
first check whether it is already wrong in the Edge screenshot of the measuring page (section 7); fix the HTML or one of
the rules above, rebuild, and look again. Do not nudge individual shapes in python-pptx: the next rebuild loses it and
the HTML and pptx drift apart.

**Check a conversion before handing it over:** open the pptx next to the Edge screenshot of the same artboard and look
at (a) the title is one line where the HTML's is, (b) each bullet's last line ends on the same word, (c) the footnote is
on the bottom rule, not under it, (d) the figure has the HTML's aspect. Any difference → the table above.

## 3. Environment — pin it, keep it project-local

The figure library decides the layout of every figure: a different matplotlib version re-runs constrained layout
differently (legends land on footnotes, titles on subtitles) even with identical data. So:

* Build figures and slides in a **project-local env** that pins the plotting stack the deck was made with — e.g.
  `<project>/<name>_env` created with `mamba create -p <project>/<name>_env -c conda-forge python=... matplotlib=<x.y.z> python-pptx=1.0 ...`.
* **Do not edit the project's own `environment.yml`, and do not install into the main `luto` env** (user rule, 2026-10-23):
  luto's versions move with the model (cuOpt pulled matplotlib to 3.11 / python 3.14), and an old matplotlib may have no
  build for its python at all.
* Find the version the existing figures were drawn with: `grep -o -m1 'Matplotlib v[0-9.]*' plots/<figure>.svg`.
  After any redraw the same grep must give the pinned version.
* An env on the Windows drive (`/mnt/f/...`) builds slowly (conda copies instead of hardlinking across the WSL/Windows
  boundary) — a few minutes for ~1.8 GB. Leave it out of any zip of the project.
* The pptx step needs Windows Edge and a Windows `node.exe` (≥ 18) — the fonts PowerPoint uses live on Windows. From WSL:
  `CMA_NODE=/mnt/f/Users/jinzhu/Documents/node-v24.13.1-win-x64/node.exe` (the env-var name in the reference
  implementation); paths are handed to Windows via `wslpath -w`.

## 4. The pipeline

| step | makes |
|---|---|
| figure script (`script/<fig>.py`) | `plots/<stem>.png` + `.svg`, `output/<stem>.csv` (the numbers the slide quotes) |
| `make_slides.py` | every artboard + `canvas.json` |
| `make_plain_deck.py [artboards...]` | the HTML deck, or with artboard names: those slides alone → `<first>.html` |
| `make_pptx.py [artboards...]` | the pptx deck, or with artboard names: those slides alone → `<first>.pptx` |

Artboards reference figures by bare file name; the builders look them up in `plots/` (and any other image dirs), and a
pass swaps a PNG reference for its SVG when one exists. A variant-figure mode (e.g. a `_shade` suffix behind an env var)
can swap in a whole alternative figure set without touching the slide text.

**The artboard generator rewrites EVERY artboard and `canvas.json` on each run.** It is deterministic, but a hand edit to
a `.dc.html` or a canvas position dragged in an editor is lost. Run it only for a deliberate rebuild, after backing up
the slides folder (the CMA project is gitignored — no history to restore from).

## 5. Writing a slide

* **Mirror an existing slide** when the new one is a variant: copy its `S["<name>"] = frame(...)` block and change only
  the text and the image name — same helper sizes, same pad / gap — so every box lands where the original's did.
* **Fit**: the artboard is a fixed 1280 × 720; overflowing text is clipped in HTML and dropped (or overlaps) in pptx.
  Rules of thumb: a size-30 title holds ≈ 60 characters on one line; a figure-left layout with the image at 440 px high
  leaves room for ~4 bullets of ~180 characters at size 15; footnotes one or two lines at size 13.
* **Quote numbers from the figure's CSV / the run's outputs**, not from memory, and say in the footnote which run or
  scenario the slide shows and how it differs from the deck's reference case.

## 6. A slide OUTSIDE the deck (variants such as `_15_percent`)

Never add a variant to `ORDER` / the titles map — that changes the deck everyone else reads. Instead:

1. **Figure** under its own suffix: give the figure script `--run <run> --tag <suffix> --label "<how titles name it>"`;
   with no arguments it must draw exactly what it drew before (output names, title text, numbers).
2. **Artboard** as a plain variable (`VARIANT = frame(...)`) written after the deck loop by its own line,
   `(OUT / "<name>_<suffix>.dc.html").write_text(...)` — not in `ORDER` / `canvas.json`. In a variant-figure mode, delete
   the copy it leaves in that mode's folder unless a variant of it is wanted.
3. **HTML** and **pptx**: `make_plain_deck.py <name>_<suffix>.dc.html`, `make_pptx.py <name>_<suffix>.dc.html`.

Before building, confirm what the variant means: run names often encode a level (`T150` = 15 %) but the letter in front
says WHICH target it applies to — several runs may be "the 15 % run".

## 7. Check, then prove the pipeline reproduces the old slide

* **Look before the pptx**: screenshot the standalone HTML with Windows Edge and read the PNG —
  ```bash
  W=$(wslpath -w "$PWD/slides/<name>.html"); O=$(wslpath -w "$PWD/slides/_measure/<name>.png")
  "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless=new --disable-gpu --hide-scrollbars \
      --window-size=1340,790 --screenshot="$O" "file:///${W//\\//}"
  ```
  Look for text running into the footnote, titles wrapping to two lines, and legends / titles colliding inside the figure.
* **Reproduce the original first** whenever a figure script or the env changes: back up the figure's PNG / SVG / CSV,
  redraw with no arguments, compare, then put the backup back —
  * PNG: pixel-identical (`numpy` arrays from PIL);
  * CSV: values identical (`pandas.DataFrame.equals`) — the bytes can differ (Windows line endings in older files);
  * SVG: identical apart from random clip-path ids and the PNG compression of embedded rasters (decode the base64
    `data:image/png` rasters and compare pixels).
  A layout difference with identical data means the wrong plotting version (section 3).
* **pptx**: build the original slide alone (`make_pptx.py <name>.dc.html`), move it to a scratch folder, and compare it
  shape by shape with its slide in the deck's pptx — shape type, left / top / width / height, text runs (text, bold,
  italic, size, font), image blob md5. A variant built the same way should then differ only in text and image content
  (boxes below shorter text move up; the tag, title and image boxes do not move).
* **Artboards vs the HTML deck**: each section of the HTML deck is an artboard body with images inlined — re-embed the
  current `.dc.html` files the same way and compare; this proves an artboard regeneration lost nothing.
* A layout bug that shows in **every** plotting version (a title `pad` too small for the subtitle under it, say) is the
  figure script's — fix the script, not the env, and note which already-built decks still carry the old figure.
