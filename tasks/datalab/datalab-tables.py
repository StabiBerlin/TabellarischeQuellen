# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anywidget==0.11.0",
#     "drawdata==0.5.2",
#     "traitlets==5.15.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import os
    from getpass import getpass
    import requests
    import time
    from io import BytesIO
    from PIL import Image
    from bs4 import BeautifulSoup
    import io
    import base64
    import anywidget
    import traitlets
    import pandas as pd
    import cv2
    import numpy as np
    import json

    return (
        BeautifulSoup,
        BytesIO,
        Image,
        anywidget,
        base64,
        cv2,
        getpass,
        io,
        mo,
        np,
        os,
        pd,
        requests,
        time,
        traitlets,
    )


@app.cell
def _(mo):
    mo.md("""
    # Tabellen mit Chandra (Datalab) extrahieren und bearbeiten

    https://documentation.datalab.to/

    **Voraussetzung**: Gültiger Datalab-Api-Key (https://www.datalab.to/app/keys)


    > In *molab* können Secrets im *developer panel* (Ctrl + J) angelegt werden, dort unter "Secrets" die Umgebungsvariable "DATALAB_API_KEY" hinzufügen.
    """)
    return


@app.cell
def _(getpass, os):
    if not os.getenv("DATALAB_API_KEY"):
        os.environ["DATALAB_API_KEY"] = getpass("Enter your Datalab API key: ")
    return


@app.cell
def _(mo):
    file_area = mo.ui.file(kind="area", filetypes=[".png", ".jpg", ".jpeg"])

    mo.vstack([mo.md("""## Eine Datei hochladen 
    Laden Sie ein Bild einer historischen Tabelle hoch (.jpg oder .png).  
        Das Notebook ist auf **einzelne Bilddateien** ausgelegt — PDFs und Mehrfachbilder 
        sind grundsätzlich möglich, aber hier nicht implementiert."""),
    file_area])
    return (file_area,)


@app.cell
def _(BytesIO, Image, file_area, mo, os):
    mo.stop(not file_area.value)
    raw_bytes = file_area.value[0].contents
    image_in = Image.open(BytesIO(raw_bytes)).convert("RGB")
    base_name = os.path.splitext(file_area.value[0].name)[0]
    return base_name, image_in


@app.cell
def _(image_in, mo):
    mo.image(src=image_in)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Bild zuschneiden / Tabelle ausschneiden
    """)
    return


@app.cell
def _(ImageCropWidget, base64, image_in, io, mo):
    _buf = io.BytesIO()
    image_in.save(_buf, format="PNG")          # PNG = lossless; use JPEG/quality=90 for large photos
    image_src = "data:image/png;base64," + base64.b64encode(_buf.getvalue()).decode()

    cropper = mo.ui.anywidget(ImageCropWidget(image_src=image_src))
    cropper
    return (cropper,)


@app.cell
def _(cropper, image_in, mo):
    crop = cropper.value.get("crop")

    if crop and crop.get("width") and crop.get("height"):
        cropped_img = image_in.crop(
            (crop["x"], crop["y"], crop["x"] + crop["width"], crop["y"] + crop["height"])
        )
    else:
        cropped_img = image_in   # noch nichts ausgewählt → ganzes Bild

    mo.image(cropped_img)
    return (cropped_img,)


@app.cell
def _(best_angle, mo):
    degree_input = mo.ui.text(label="Drehen (Grad)", value="0")
    mo.vstack([
        mo.md("""### Bild drehen / Tabelle ausrichten
        Eine schiefe Tabelle verschlechtert die Erkennungsqualität erheblich.  
        Der vorgeschlagene Winkel wird anhand der horizontalen Strukturen im Bild berechnet 
        (Projektionsprofilanalyse). Passen Sie den Wert bei Bedarf manuell an und prüfen Sie 
        das Ergebnis in der Vorschau."""), 
        mo.hstack([degree_input, mo.md(f"Vorschlag: {-best_angle:.2f} Grad")])])
    return (degree_input,)


@app.cell
def _(Image, cropped_img, degree_input, mo, np):
    def parse_angle(raw: str) -> float:
        try:
            return float(raw.strip())
        except (ValueError, AttributeError):
            return 0.0

    angle = parse_angle(degree_input.value)

    def rotate_image(img, angle: float):
        angle = angle % 360
        coarse = round(angle / 90) * 90
        fine = angle - coarse
        coarse = coarse % 360

        # Step 1: lossless coarse rotation (modern enum constants)
        T = Image.Transpose
        if coarse == 90:
            result = img.transpose(T.ROTATE_270)
        elif coarse == 180:
            result = img.transpose(T.ROTATE_180)
        elif coarse == 270:
            result = img.transpose(T.ROTATE_90)
        else:
            result = img

        # Step 2: fine correction
        if fine != 0:
            result = result.rotate(-fine, resample=Image.BICUBIC, expand=False)

        return result

    image = rotate_image(cropped_img, angle)   # <-- was image_in

    mo.image(src=np.asarray(image))
    return (image,)


@app.cell
def _(cv2, image_in, np):
    def find_skew_by_projection(angle_range=3.0, step=0.1):
        img = cv2.cvtColor(np.array(image_in), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        best_angle = 0
        best_score = -1

        for angle in np.arange(-angle_range, angle_range + step, step):
            (h, w) = thresh.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_NEAREST)

            # Sum pixel values across each row
            row_sums = np.sum(rotated, axis=1)

            # Score = variance of row sums — higher variance means rows are
            # crisply separated (well-aligned rules/text lines), lower variance
            # means everything is smeared together (misaligned)
            score = np.var(row_sums)

            if score > best_score:
                best_score = score
                best_angle = angle

        return best_angle

    best_angle = find_skew_by_projection()
    print(f"Best angle: {best_angle:.2f} degrees")
    return (best_angle,)


@app.cell
def _(image, mo):
    mo.stop(not image)
    run_button = mo.ui.run_button(label="Bild prozessieren!")
    mo.vstack([mo.md("""### Tabelle erkennen lassen (API-Aufruf)
                        Das Bild wird jetzt an die API von Datalab gesendet.  
                        Die Verarbeitung dauert typischerweise **10–30 Sekunden**.  
                        Kosten: abhängig von Seitenzahl und Modus — wird nach dem Aufruf angezeigt."""),
              run_button])
    return (run_button,)


@app.cell
def _(BytesIO, image, mo, os, requests, run_button, time):
    mo.stop(not run_button.value)
    url = "https://www.datalab.to/api/v1/convert"
    headers = {"X-API-Key": os.environ["DATALAB_API_KEY"]}


    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    buf.seek(0)

    response = requests.post(
        url,
        files={"file": ("image.jpg", buf, "image/jpeg")},
        data={
            "output_format": "json,html",
            "mode": "accurate",
            "extras": "table_cell_bboxes",
            #"use_llm": "true",
            #"skip_cache": "true",
            #"word_bboxes": "true",
            #"add_block_ids": "true"

        },
        headers=headers,
    )

    data = response.json()
    check_url = data["request_check_url"]
    print(data)

    with mo.status.spinner(title="arbeitet...") as spinner:
        # Poll until finished
        while True:
            r = requests.get(check_url, headers=headers)
            result = r.json()

            if result.get("status") == "complete":
                print(f"Kosten: {result.get("cost_breakdown", {}).get("final_cost_cents")} cent")
                resultHTML = result.get("html")
                resultJSON = result.get("json")
                break

            if result.get("error"):
                raise RuntimeError(result["error"])

            time.sleep(2)
    return result, resultHTML, resultJSON


@app.cell
def _(mo, result):
    cost_display = mo.callout(
            mo.md(f"**API-Kosten für diesen Aufruf:** {result.get("cost_breakdown", {}).get("final_cost_cents")} cent"),
            kind="info"
        ) if {result.get("cost_breakdown", {}).get("final_cost_cents")} is not None else None

    cost_display
    return


@app.cell
def _(image, resultJSON):
    page_block = resultJSON["children"][0]  # the top-level 'Page' block
    ref_x1, ref_y1, ref_x2, ref_y2 = page_block["bbox"]
    ref_w = ref_x2 - ref_x1
    ref_h = ref_y2 - ref_y1


    scale_x = image.width / ref_w
    scale_y = image.height / ref_h
    print(f"image: {image.width}x{image.height}, API: {ref_w}x{ref_h}, scale: {scale_x:.4f}, {scale_y:.4f}")
    return scale_x, scale_y


@app.cell
def _():
    #resultJSON
    return


@app.cell
def _(Image, base64, image, io, mo, resultHTML):
    styled_html = f"""
    <style>
    table {{
        border-collapse: collapse;
    }}
    th, td {{
        border: 1px solid #444;
        padding: 4px 8px;
    }}
    </style>

    {resultHTML}
    """

    def pil_to_data_url_resized(image, max_width=1200):
        ratio = min(1.0, max_width / image.width)
        display_size = (int(image.width * ratio), int(image.height * ratio))
        display_image = image.resize(display_size, Image.LANCZOS)
        buffer = io.BytesIO()
        display_image.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    display_image_url = pil_to_data_url_resized(image)

    mo.vstack([mo.md("""### Transkriptionsergebnis der gesamten Seite"""),    
    mo.Html(f"""
    <div style="display:flex; gap:16px; align-items:flex-start;">
        <div style="flex:1; height:600px; overflow:auto; min-width:0;">
            <img src="{display_image_url}" style="max-width:none;">
        </div>
        <div style="flex:1; height:600px; overflow:auto; min-width:0;">
            {styled_html}
        </div>
    </div>
    """)])
    return (pil_to_data_url_resized,)


@app.cell
def _(BeautifulSoup, mo, resultJSON):
    def extract_tables(resultJSON):
        tables = []
        for page in resultJSON["children"]:
            for element in page["children"]:
                if element["block_type"] == "Table":
                    tables.append(element)
                elif element["block_type"] == "ListGroup":
                    # Table wrapped in <ul><li>...</li></ul>
                    soup = BeautifulSoup(element["html"], "html.parser")
                    for table_tag in soup.find_all("table"):
                        # Synthesize a table-like dict with the same shape
                        # your downstream code expects
                        tables.append({
                            "block_type": "Table",
                            "html": str(table_tag),
                            "bbox": element["bbox"],
                            "page": element["page"],
                        })
                elif element["block_type"] == "Text":
                    # Table wrapped in <ul><li>...</li></ul>
                    soup = BeautifulSoup(element["html"], "html.parser")
                    for table_tag in soup.find_all("table"):
                        # Synthesize a table-like dict with the same shape
                        # your downstream code expects
                        tables.append({
                            "block_type": "Table",
                            "html": str(table_tag),
                            "bbox": element["bbox"],
                            "page": element["page"],
                        })
        return tables

    tables = extract_tables(resultJSON)

    table_options = {
        f"Tabelle {i + 1}": i
        for i in range(len(tables))
    }

    if len(tables)>0:
        selected_table_idx = mo.ui.dropdown(
            options=table_options,
            value="Tabelle 1",
            label="Tabelle zur Bearbeitung auswählen",
        )


    mo.vstack([mo.md(f"### Es wurde(n) {len(tables)} Tabelle(n) gefunden."),
              selected_table_idx]) if len(tables)>0 else mo.md(f"Es wurden keine Tabellen gefunden.")
    return selected_table_idx, tables


@app.cell
def _(CellBBoxViewer, base64, cell_bboxes, image, io, mo):
    def pil_to_data_url(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return f"data:image/png;base64,{encoded}"

    page_image_url = pil_to_data_url(image)

    bbox_viewer = CellBBoxViewer(
        image=page_image_url,
        boxes=cell_bboxes,
    )


    mo.vstack([mo.md("""### Tabellenlayout (bounding boxes), die Chandra erkannt hat
                            Die Tabellenfarbe richtet sich nach der Transkriptionskonfidenz
                            - <span style="color:#dc2626">■</span> Rot < 0.80
                            - <span style="color:#f59e0b">■</span> Orange = 0.80–0.95
                            - <span style="color:#2563eb">■</span> Blau > 0.95
                            """),
    bbox_viewer])
    return


@app.cell
def _(mo, selected_table_idx, tables):
    original_tables = {i: t["html"] for i, t in enumerate(tables)}
    get_saved_tables, set_saved_tables = mo.state({i: t["html"] for i, t in enumerate(tables)})
    mo.Html(get_saved_tables()[selected_table_idx.value])
    return get_saved_tables, original_tables, set_saved_tables


@app.cell
def _(BeautifulSoup, get_saved_tables, scale_x, scale_y, selected_table_idx):
    def extract_cell_boxes(table_html: str, scale_x: float, scale_y: float) -> list[dict]:
            soup = BeautifulSoup(table_html, "html.parser")
            boxes = []
            for row_idx, tr in enumerate(soup.find_all("tr")):
                for col_idx, cell in enumerate(tr.find_all(["td", "th"])):
                    bbox_attr = cell.get("data-bbox")
                    if not bbox_attr:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in bbox_attr.split())
                    boxes.append({
                        "x1": x1 * scale_x, "y1": y1 * scale_y,
                        "x2": x2 * scale_x, "y2": y2 * scale_y,
                        "label": f"r{row_idx}c{col_idx}",
                        "text": cell.get_text(" ", strip=True),
                        "confidence": float(cell.get("data-confidence", 1.0)),
                        "colspan": int(cell.get("colspan", 1)),
                        "rowspan": int(cell.get("rowspan", 1)),
                    })
            return boxes



    cell_bboxes = extract_cell_boxes(
        get_saved_tables()[selected_table_idx.value],
        scale_x,
        scale_y,
    )
    return (cell_bboxes,)


@app.cell
def _(mo, tables):
    mo.md("""### Struktur der Tabelle korrigieren
        Hier können Sie die **Struktur** der Tabelle anpassen — also Zellen zusammenführen, 
        trennen, Zeilen oder Spalten löschen.  
        Tun Sie dies **vor** der Textkorrektur, da Strukturänderungen den Text überschreiben können.

        > Zellen auswählen: Klick (einzeln) oder Shift+Klick (mehrere). Dann die gewünschte Aktion wählen.  
        > **Wichtig: Änderungen erst mit „Save changes" bestätigen**, bevor Sie weitermachen.""") if tables else None
    return


@app.cell
def _(
    TableStructureEditor,
    get_saved_tables,
    image,
    mo,
    pil_to_data_url_resized,
    scale_x,
    scale_y,
    selected_table_idx,
    tables,
):
    _current_html = get_saved_tables()[selected_table_idx.value]
    _raw_structure_editor = TableStructureEditor(
        html=_current_html,
        value=_current_html,
    )

    structure_editor = mo.ui.anywidget(
        _raw_structure_editor
    )

    tx1, ty1, tx2, ty2 = tables[selected_table_idx.value]["bbox"]
    cropped = image.crop((tx1 * scale_x, ty1 * scale_y, tx2 * scale_x, ty2 * scale_y))
    cropped_url = pil_to_data_url_resized(cropped)


    left = mo.Html(f"""
    <div style="height:600px; overflow:auto;">
        <img src="{cropped_url}" style="max-width:none;">
    </div>
    """)

    mo.hstack([left, structure_editor]) if cropped.width < 1500 else mo.vstack([
        mo.Html(f'<img src="{cropped_url}" style="width:100%;">'),
        structure_editor
    ])
    return cropped, cropped_url, left, structure_editor


@app.cell
def _(
    get_saved_tables,
    selected_table_idx,
    set_saved_tables,
    structure_editor,
):
    _new_value = structure_editor.value["value"]
    if get_saved_tables().get(selected_table_idx.value) != _new_value:
        set_saved_tables(lambda old: {**old, selected_table_idx.value: _new_value})
    return


@app.cell
def _(mo, tables):
    mo.md("""
            ### Text korrigieren
            Jetzt können Sie den **transkribierten Text** direkt in der Tabelle bearbeiten —  
            einfach in eine Zelle klicken und tippen.
            > **Wichtig: Auch hier erst „Save changes" klicken**, bevor Sie exportieren.""") if tables else None
    return


@app.cell
def _(
    EditableHTMLTable,
    cropped,
    cropped_url,
    get_saved_tables,
    left,
    mo,
    selected_table_idx,
):
    _current_html = get_saved_tables()[selected_table_idx.value]
    _raw_editor = EditableHTMLTable(
        html=_current_html,
        value=_current_html,
        )



    editor = mo.ui.anywidget(_raw_editor)


    mo.hstack([left, editor]) if cropped.width < 1500 else mo.vstack([
        mo.Html(f'<img src="{cropped_url}" style="width:100%;">'),
        editor
    ])
    return (editor,)


@app.cell
def _(editor, get_saved_tables, selected_table_idx, set_saved_tables):
    _new_value = editor.value["value"]
    if get_saved_tables().get(selected_table_idx.value) != _new_value:
        set_saved_tables(lambda old: {**old, selected_table_idx.value: _new_value})
    return


@app.cell
def _(editor, mo):
    edited_html = editor.value["value"]

    mo.vstack([mo.md("""### Überarbeitete Fassung der Tabelle"""), mo.Html(edited_html)])
    return (edited_html,)


@app.cell
def _(edited_html, io, mo, pd):
    # Convert HTML table to pandas DataFrame
    df =  pd.read_html(io.StringIO(edited_html))[0]

    mo.vstack([mo.md("""### Versuch, die Tabelle in einen Dataframe (pandas) zu parsen"""),
    df])
    return


@app.cell
def _(BeautifulSoup):
    def build_edited_full_html(page_html: str, saved_tables: dict[int, str]) -> str:
        soup = BeautifulSoup(page_html, "html.parser")
        original_tables = soup.find_all("table")

        for i, table_tag in enumerate(original_tables):
            if i not in saved_tables:
                continue
            replacement_soup = BeautifulSoup(saved_tables[i], "html.parser")
            replacement_table = replacement_soup.find("table")
            if replacement_table is not None:
                table_tag.replace_with(replacement_table)

        return str(soup)

    return (build_edited_full_html,)


@app.cell
def _(
    build_edited_full_html,
    get_saved_tables,
    mo,
    resultHTML,
    selected_table_idx,
):
    full_html = build_edited_full_html(resultHTML, get_saved_tables())
    single_table_html = get_saved_tables()[selected_table_idx.value]

    mo.vstack([
        mo.md("""### Download-Optionen
                    (.csv, .json) können direkt aus dem Dataframe exportiert werden (s.o. "Export")"""),
        mo.hstack([
            mo.download(
                data=full_html.encode("utf-8"),
                filename="full_page_edited.html",
                mimetype="text/html",
                label="Gesamte Seite (inkl. aller Tabellen) herunterladen (Html)"),
            mo.download(
                data=single_table_html.encode("utf-8"),
                filename=f"table_{selected_table_idx.value + 1}.html",
                mimetype="text/html",
                label="Nur die ausgewählte Tabelle herunterladen (Html)")
        ])
    ])
  
    return


@app.cell
def _(BeautifulSoup):
    def bbox_to_points(x1, y1, x2, y2):
        return f"{int(x1)},{int(y1)} {int(x2)},{int(y1)} {int(x2)},{int(y2)} {int(x1)},{int(y2)}"

    def _append_table_region(page, html, table_idx, table_bbox, scale_x, scale_y):
        from xml.etree.ElementTree import SubElement

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return

        tx1, ty1, tx2, ty2 = table_bbox
        table_id = f"table_{table_idx}"

        table_region = SubElement(page, "TableRegion")
        table_region.set("id", table_id)
        table_region.set("lineSeparators", "true")

        coords = SubElement(table_region, "Coords")
        coords.set("points", bbox_to_points(
            tx1 * scale_x, ty1 * scale_y,
            tx2 * scale_x, ty2 * scale_y
        ))

        rows = table.find_all("tr")
        grid = {}
        cell_id = 0
        parsed_cells = []

        for row_idx, tr in enumerate(rows):
            col_idx = 0
            for cell in tr.find_all(["td", "th"]):
                # Skip positions already occupied by a rowspan/colspan from an earlier cell
                while grid.get((row_idx, col_idx)):
                    col_idx += 1

                colspan = int(cell.get("colspan", 1))
                rowspan = int(cell.get("rowspan", 1))
                is_header = cell.name == "th"

                for r in range(row_idx, row_idx + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        grid[(r, c)] = True

                bbox_attr = cell.get("data-bbox")
                if bbox_attr:
                    bx1, by1, bx2, by2 = (float(v) for v in bbox_attr.split())
                    bx1 *= scale_x
                    by1 *= scale_y
                    bx2 *= scale_x
                    by2 *= scale_y
                else:
                    bx1 = tx1 * scale_x
                    by1 = ty1 * scale_y
                    bx2 = tx2 * scale_x
                    by2 = ty2 * scale_y

                parsed_cells.append({
                    "row": row_idx,
                    "col": col_idx,
                    "rowspan": rowspan,
                    "colspan": colspan,
                    "is_header": is_header,
                    "bbox": (bx1, by1, bx2, by2),
                    "text": cell.get_text(" ", strip=True),
                    "id": f"{table_id}_cell_{cell_id}",
                })

                cell_id += 1
                col_idx += colspan

        if parsed_cells:
            # Account for spans when computing the true row/column extent
            num_rows = max(c["row"] + c["rowspan"] for c in parsed_cells)
            num_cols = max(c["col"] + c["colspan"] for c in parsed_cells)
            table_region.set("rows", str(num_rows))
            table_region.set("columns", str(num_cols))

        y_boundaries = set()
        x_boundaries = set()
        for c in parsed_cells:
            bx1, by1, bx2, by2 = c["bbox"]
            y_boundaries.add(by1)
            y_boundaries.add(by2)
            x_boundaries.add(bx1)
            x_boundaries.add(bx2)

        sorted_y = sorted(y_boundaries)
        sorted_x = sorted(x_boundaries)

        # The schema requires at least 2 GridPoints rows; fall back to the table
        # boundary if cell data alone doesn't provide two distinct y values
        if len(sorted_y) < 2:
            sorted_y = sorted({ty1 * scale_y, ty2 * scale_y} | set(sorted_y))

        grid_el = SubElement(table_region, "Grid")
        for i, y in enumerate(sorted_y):
            gp = SubElement(grid_el, "GridPoints")
            gp.set("index", str(i))
            # One point per x boundary at this y level, encoding the full column structure
            points = " ".join(f"{int(x)},{int(y)}" for x in sorted_x)
            gp.set("points", points)

        # Cell TextRegions are siblings of TableRegion under Page, not children of it.
        # The custom attribute links each cell back to its parent table, following the
        # convention used by tools such as Aletheia and PRImA's PAGE libraries.
        for c in parsed_cells:
            bx1, by1, bx2, by2 = c["bbox"]

            text_region = SubElement(page, "TextRegion")
            text_region.set("id", c["id"])
            text_region.set("custom", f"tableId:{table_id}")

            cell_coords = SubElement(text_region, "Coords")
            cell_coords.set("points", bbox_to_points(bx1, by1, bx2, by2))

            roles = SubElement(text_region, "Roles")
            cell_role = SubElement(roles, "TableCellRole")
            cell_role.set("rowIndex", str(c["row"]))
            cell_role.set("columnIndex", str(c["col"]))
            # rowSpan/colSpan default to 1 in the schema, so omit them when not needed
            if c["rowspan"] != 1:
                cell_role.set("rowSpan", str(c["rowspan"]))
            if c["colspan"] != 1:
                cell_role.set("colSpan", str(c["colspan"]))
            if c["is_header"]:
                cell_role.set("header", "true")

            if c["text"]:
                text_line = SubElement(text_region, "TextLine")
                text_line.set("id", f"{c['id']}_line")
                line_coords = SubElement(text_line, "Coords")
                line_coords.set("points", bbox_to_points(bx1, by1, bx2, by2))
                baseline = SubElement(text_line, "Baseline")
                baseline.set("points", f"{int(bx1)},{int(by2)} {int(bx2)},{int(by2)}")
                text_equiv = SubElement(text_line, "TextEquiv")
                SubElement(text_equiv, "Unicode").text = c["text"]

                region_text_equiv = SubElement(text_region, "TextEquiv")
                SubElement(region_text_equiv, "Unicode").text = c["text"]


    def html_page_to_page_xml(
        saved_tables: dict,
        tables: list,
        image_filename: str,
        image_width: int,
        image_height: int,
        scale_x: float,
        scale_y: float,
    ) -> str:
        from xml.etree.ElementTree import Element, SubElement, tostring
        from xml.dom import minidom
        import datetime

        NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
        XSI = "http://www.w3.org/2001/XMLSchema-instance"
        SCHEMA_LOC = (
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 "
            "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
        )

        root = Element("PcGts")
        root.set("xmlns", NS)
        root.set("xmlns:xsi", XSI)
        root.set("xsi:schemaLocation", SCHEMA_LOC)

        meta = SubElement(root, "Metadata")
        SubElement(meta, "Creator").text = "Datalab Table Extractor"
        SubElement(meta, "Created").text = datetime.datetime.now().isoformat()
        SubElement(meta, "LastChange").text = datetime.datetime.now().isoformat()

        page = SubElement(root, "Page")
        page.set("imageFilename", image_filename)
        page.set("imageWidth", str(image_width))
        page.set("imageHeight", str(image_height))

        for table_idx, table in enumerate(tables):
            html = saved_tables.get(table_idx, table["html"])
            tx1, ty1, tx2, ty2 = table["bbox"]
            _append_table_region(
                page=page,
                html=html,
                table_idx=table_idx,
                table_bbox=(tx1, ty1, tx2, ty2),
                scale_x=scale_x,
                scale_y=scale_y,
            )

        xml_str = tostring(root, encoding="unicode")
        return minidom.parseString(xml_str).toprettyxml(indent="  ")

    return (html_page_to_page_xml,)


@app.cell
def _(
    base64,
    base_name,
    file_area,
    get_saved_tables,
    html_page_to_page_xml,
    image,
    mo,
    scale_x,
    scale_y,
    tables,
):
    full_xml = html_page_to_page_xml(
        saved_tables=get_saved_tables(),
        tables=tables,
        image_filename=file_area.value[0].name,
        image_width=image.width,
        image_height=image.height,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    b64 = base64.b64encode(full_xml.encode("utf-8")).decode("utf-8")
    mo.Html(f'''
    <a download="{base_name}.xml" 
       href="data:application/xml;base64,{b64}"
       style="
           display: inline-flex;
           align-items: center;
           padding: 4px 12px;
           background: white;
           color: #111;
           border: 1px solid #d1d5db;
           border-radius: 6px;
           font-size: 14px;
           font-family: inherit;
           text-decoration: none;
           cursor: pointer;
       ">
       Als PAGE-XML herunterladen
    </a>
    ''')
    return


@app.cell
def _(mo, original_tables, selected_table_idx, set_saved_tables, tables):
    mo.stop(not tables)
    revert_button = mo.ui.button(
        label="↶ Alle Änderungen an der Tabelle zurücksetzen",
        on_click=lambda _: set_saved_tables(lambda old: {**old, selected_table_idx.value: original_tables[selected_table_idx.value]}),
        kind="danger",
    )
    revert_button
    return


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class CellBBoxViewer(anywidget.AnyWidget):
        image = traitlets.Unicode("").tag(sync=True)
        boxes = traitlets.List([]).tag(sync=True)
        selected = traitlets.Unicode("").tag(sync=True)

        _esm = """
        export default {
            render({ model, el }) {
                el.innerHTML = "";

                // --------------------------------------------------
                // Container
                // --------------------------------------------------

                const container = document.createElement("div");

                Object.assign(container.style, {
                    position: "relative",
                    width: "100%",
                    overflow: "auto",
                    border: "1px solid #d1d5db",
                    borderRadius: "6px",
                    background: "#f3f4f6",
                    lineHeight: "0"
                });

                // --------------------------------------------------
                // Image
                // --------------------------------------------------

                const image = document.createElement("img");

                image.src = model.get("image");

                Object.assign(image.style, {
                    display: "block",
                    width: "100%",
                    height: "auto",
                    margin: "0",
                    padding: "0",
                    border: "0"
                });

                // --------------------------------------------------
                // SVG overlay
                // --------------------------------------------------

                const overlay = document.createElementNS(
                    "http://www.w3.org/2000/svg",
                    "svg"
                );

                Object.assign(overlay.style, {
                    position: "absolute",
                    top: "0",
                    left: "0",
                    width: "100%",
                    height: "100%",
                    margin: "0",
                    padding: "0",
                    border: "0",
                    display: "block",
                    pointerEvents: "none"
                });

                // Important:
                // Make the SVG use exactly the same aspect ratio as
                // the image. Otherwise SVG's default preserveAspectRatio
                // behavior can introduce a small translation/scale
                // difference.
                overlay.setAttribute(
                    "preserveAspectRatio",
                    "none"
                );

                // --------------------------------------------------
                // Helpers
                // --------------------------------------------------

                function confidenceColor(confidence) {
                    if (confidence < 0.80) {
                        return "#dc2626";
                    }

                    if (confidence < 0.95) {
                        return "#f59e0b";
                    }

                    return "#2563eb";
                }

                // --------------------------------------------------
                // Draw
                // --------------------------------------------------

                function draw() {
                    overlay.innerHTML = "";

                    const boxes = model.get("boxes") || [];

                    const naturalWidth = image.naturalWidth;
                    const naturalHeight = image.naturalHeight;

                    if (!naturalWidth || !naturalHeight) {
                        return;
                    }

                    // SVG coordinate system is exactly the native
                    // pixel coordinate system of the image.
                    overlay.setAttribute(
                        "viewBox",
                        `0 0 ${naturalWidth} ${naturalHeight}`
                    );

                    // Explicitly synchronize the SVG's rendered size
                    // with the image's rendered size.
                    const renderedWidth = image.getBoundingClientRect().width;
                    const renderedHeight = image.getBoundingClientRect().height;

                    overlay.style.width = `${renderedWidth}px`;
                    overlay.style.height = `${renderedHeight}px`;

                    for (const box of boxes) {
                        const color =
                            confidenceColor(
                                Number(box.confidence ?? 1)
                            );

                        const rect =
                            document.createElementNS(
                                "http://www.w3.org/2000/svg",
                                "rect"
                            );

                        rect.setAttribute("x", box.x1);
                        rect.setAttribute("y", box.y1);

                        rect.setAttribute(
                            "width",
                            box.x2 - box.x1
                        );

                        rect.setAttribute(
                            "height",
                            box.y2 - box.y1
                        );

                        rect.setAttribute(
                            "fill",
                            color
                        );

                        rect.setAttribute(
                            "fill-opacity",
                            "0.08"
                        );

                        rect.setAttribute(
                            "stroke",
                            color
                        );

                        rect.setAttribute(
                            "stroke-width",
                            "2"
                        );

                        rect.style.pointerEvents = "auto";
                        rect.style.cursor = "pointer";

                        rect.addEventListener("click", () => {
                            model.set(
                                "selected",
                                box.label
                            );

                            model.save_changes();
                        });

                        overlay.appendChild(rect);
                    }
                }

                // --------------------------------------------------
                // Keep overlay synchronized with image
                // --------------------------------------------------

                image.addEventListener("load", draw);

                // Redraw when the browser resizes the displayed image.
                const resizeObserver = new ResizeObserver(() => {
                    draw();
                });

                resizeObserver.observe(image);

                // --------------------------------------------------
                // Assemble
                // --------------------------------------------------

                container.appendChild(image);
                container.appendChild(overlay);

                el.appendChild(container);

                if (image.complete) {
                    draw();
                }
            }
        };
    """

    return (CellBBoxViewer,)


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class TableStructureEditor(anywidget.AnyWidget):
        html = traitlets.Unicode("").tag(sync=True)
        value = traitlets.Unicode("").tag(sync=True)

        _esm = """
        export default {
            render({ model, el }) {

                el.innerHTML = "";

                // --------------------------------------------------
                // Layout
                // --------------------------------------------------

                const wrapper = document.createElement("div");

                Object.assign(wrapper.style, {
                    width: "100%",
                    fontFamily: "system-ui, sans-serif"
                });


                // --------------------------------------------------
                // Toolbar
                // --------------------------------------------------

                const toolbar = document.createElement("div");

                Object.assign(toolbar.style, {
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    marginBottom: "12px",
                    flexWrap: "wrap"
                });


                function makeButton(text) {
                    const button =
                        document.createElement("button");

                    button.textContent = text;

                    Object.assign(button.style, {
                        padding: "7px 12px",
                        border: "1px solid #9ca3af",
                        borderRadius: "6px",
                        background: "white",
                        color: "#374151",
                        fontSize: "14px",
                        fontWeight: "600",
                        cursor: "pointer"
                    });

                    return button;
                }


                const mergeButton =
                    makeButton("Merge selected");

                const splitButton =
                    makeButton("Split cell");

                const saveButton =
                    makeButton("✓  Save changes");

                const revertButton =
                    makeButton("↶  Revert");

                const deleteRowBtn = makeButton("Delete row");
                    toolbar.appendChild(deleteRowBtn);

                const deleteColBtn = makeButton("Delete column");
                    toolbar.appendChild(deleteColBtn);

                // --------------------------------------------------
                // Table area
                // --------------------------------------------------

                const tableContainer =
                    document.createElement("div");

                Object.assign(tableContainer.style, {
                    width: "100%",
                    overflowX: "auto"
                });


                // --------------------------------------------------
                // Selection state
                // --------------------------------------------------

                const selectedCells = new Set();


                function updateSelectionStyles() {

                    const table =
                        tableContainer.querySelector("table");

                    if (!table) {
                        return;
                    }



                    table
                        .querySelectorAll("td, th")
                        .forEach((cell) => {

                            if (selectedCells.has(cell)) {

                                cell.style.outline =
                                    "3px solid #2563eb";

                                cell.style.outlineOffset =
                                    "-3px";

                                cell.style.backgroundColor =
                                    "rgba(37, 99, 235, 0.12)";

                            } else {

                                cell.style.outline = "";
                                cell.style.backgroundColor = "";
                            }
                        });
                }


                function clearSelection() {
                    selectedCells.clear();
                    updateSelectionStyles();
                }


                // --------------------------------------------------
                // Build selected table
                // --------------------------------------------------

                function buildTable(html) {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, "text/html");
                    const table = doc.querySelector("table");

                    if (!table) {
                        tableContainer.textContent = "No table found.";
                        return null;
                    }

                     table.style.borderCollapse = "collapse";
                        table.querySelectorAll("td, th").forEach(cell => {
                            cell.style.border = "1px solid #444";
                            cell.style.padding = "4px 8px";
                        });

                    table
                        .querySelectorAll("td, th")
                        .forEach((cell) => {

                            cell.style.cursor = "pointer";

                            cell.addEventListener(
                                "click",
                                (event) => {

                                    event.preventDefault();

                                    if (event.shiftKey) {

                                        if (
                                            selectedCells.has(
                                                cell
                                            )
                                        ) {
                                            selectedCells.delete(
                                                cell
                                            );
                                        } else {
                                            selectedCells.add(
                                                cell
                                            );
                                        }

                                    } else {

                                        selectedCells.clear();

                                        selectedCells.add(
                                            cell
                                        );
                                    }

                                    updateSelectionStyles();
                                }
                            );
                        });


                    tableContainer.innerHTML = "";
                    tableContainer.appendChild(table);

                    updateSelectionStyles();

                    return table;
                }


                // --------------------------------------------------
                // Read current table
                // --------------------------------------------------

                function getCurrentTableHTML() {

                    const table =
                        tableContainer.querySelector(
                            "table"
                        );

                    return table
                        ? table.outerHTML
                        : "";
                }


                // --------------------------------------------------
                // Load saved HTML
                // --------------------------------------------------

                function loadSavedHTML() {

                    selectedCells.clear();

                    buildTable(
                        model.get("value") ||
                        model.get("html")
                    );
                }


                // --------------------------------------------------
                // DELETE ROW
                // --------------------------------------------------

                deleteRowBtn.addEventListener("click", () => {
                    if (selectedCells.size === 0) return;

                    // Collect unique rows from selected cells
                    const rows = new Set(
                        Array.from(selectedCells).map(cell => cell.parentElement)
                    );

                    rows.forEach(row => row.remove());

                    clearSelection();
                });


                // --------------------------------------------------
                // DELETE COL
                // --------------------------------------------------

                deleteColBtn.addEventListener("click", () => {
                    if (selectedCells.size === 0) return;

                    // Collect unique column indices from selected cells
                    const colIndices = new Set(
                        Array.from(selectedCells).map(cell => cell.cellIndex)
                    );

                    const table = tableContainer.querySelector("table");
                    if (!table) return;

                    // Delete from right to left to avoid index shifting
                    const sortedIndices = Array.from(colIndices).sort((a, b) => b - a);

                    table.querySelectorAll("tr").forEach(row => {
                        sortedIndices.forEach(idx => {
                            const cell = row.cells[idx];
                            if (cell) cell.remove();
                        });
                    });

                    clearSelection();
                });


                // --------------------------------------------------
                // MERGE
                // --------------------------------------------------

                mergeButton.addEventListener("click", () => {
                    if (selectedCells.size < 2) return;

                    const cells = Array.from(selectedCells);

                    // Sort top-to-bottom, left-to-right
                    cells.sort((a, b) => {
                        const rowA = a.parentElement ? a.parentElement.rowIndex : 0;
                        const rowB = b.parentElement ? b.parentElement.rowIndex : 0;
                        if (rowA !== rowB) return rowA - rowB;
                        return a.cellIndex - b.cellIndex;
                    });

                    const first = cells[0];

                    // Determine merged rectangle
                    let minRow = Infinity, maxRow = -Infinity;
                    let minCol = Infinity, maxCol = -Infinity;

                    for (const cell of cells) {
                        const row = cell.parentElement ? cell.parentElement.rowIndex : 0;
                        const col = cell.cellIndex;
                        const rowspan = parseInt(cell.getAttribute("rowspan") || "1");
                        const colspan = parseInt(cell.getAttribute("colspan") || "1");

                        minRow = Math.min(minRow, row);
                        maxRow = Math.max(maxRow, row + rowspan - 1);
                        minCol = Math.min(minCol, col);
                        maxCol = Math.max(maxCol, col + colspan - 1);
                    }

                    // Merge bbox
                    const bboxes = cells.map(cell => {
                        const attr = cell.getAttribute("data-bbox");
                        if (!attr) return null;
                        const [x1, y1, x2, y2] = attr.split(" ").map(Number);
                        return { x1, y1, x2, y2 };
                    }).filter(Boolean);

                    if (bboxes.length > 0) {
                        const merged = {
                            x1: Math.min(...bboxes.map(b => b.x1)),
                            y1: Math.min(...bboxes.map(b => b.y1)),
                            x2: Math.max(...bboxes.map(b => b.x2)),
                            y2: Math.max(...bboxes.map(b => b.y2)),
                        };
                        first.setAttribute("data-bbox",
                            `${merged.x1} ${merged.y1} ${merged.x2} ${merged.y2}`);
                    }

                    // Merge content
                    const contents = cells
                        .map(cell => cell.innerHTML.trim())
                        .filter(text => text.length > 0);

                    first.innerHTML = contents.join("");

                    // Set colspan/rowspan
                    const mergedRowspan = maxRow - minRow + 1;
                    const mergedColspan = maxCol - minCol + 1;

                    if (mergedRowspan > 1) {
                        first.setAttribute("rowspan", String(mergedRowspan));
                    } else {
                        first.removeAttribute("rowspan");
                    }

                    if (mergedColspan > 1) {
                        first.setAttribute("colspan", String(mergedColspan));
                    } else {
                        first.removeAttribute("colspan");
                    }

                    // Remove other selected cells
                    for (const cell of cells.slice(1)) {
                        cell.remove();
                    }

                    clearSelection();
                });

                // --------------------------------------------------
                // SPLIT
                // --------------------------------------------------

                    splitButton.addEventListener("click", () => {
                        if (selectedCells.size !== 1) return;

                        const cell = Array.from(selectedCells)[0];
                        const row = cell.parentElement;
                        const colspan = parseInt(cell.getAttribute("colspan") || "1");

                        // Split colspan evenly between the two resulting cells
                        const newColspan = Math.floor(colspan / 2);
                        const remainder = colspan - newColspan;

                        if (remainder > 1) {
                            cell.setAttribute("colspan", String(remainder));
                        } else {
                            cell.removeAttribute("colspan");
                        }

                        const newCell = document.createElement(cell.tagName);
                        if (newColspan > 1) {
                            newCell.setAttribute("colspan", String(newColspan));
                        }

                        // Split bbox at midpoint
                        const attr = cell.getAttribute("data-bbox");
                        if (attr) {
                            const [x1, y1, x2, y2] = attr.split(" ").map(Number);
                            const midX = (x1 + x2) / 2;
                            cell.setAttribute("data-bbox", `${x1} ${y1} ${midX} ${y2}`);
                            newCell.setAttribute("data-bbox", `${midX} ${y1} ${x2} ${y2}`);
                        }

                        cell.insertAdjacentElement("afterend", newCell);
                        clearSelection();
                    });


                // --------------------------------------------------
                // SAVE
                // --------------------------------------------------

                saveButton.addEventListener(
                    "click",
                    () => {

                        const currentHTML =
                            getCurrentTableHTML();

                        if (!currentHTML) {
                            return;
                        }

                        model.set(
                            "value",
                            currentHTML
                        );

                        model.save_changes();

                        saveButton.textContent =
                            "✓  Saved!";

                        setTimeout(
                            () => {
                                saveButton.textContent =
                                    "✓  Save changes";
                            },
                            1000
                        );
                    }
                );


                // --------------------------------------------------
                // REVERT
                // --------------------------------------------------

                revertButton.addEventListener(
                    "click",
                    () => {
                        loadSavedHTML();
                    }
                );


                // --------------------------------------------------
                // Assemble
                // --------------------------------------------------

                toolbar.appendChild(
                    mergeButton
                );

                toolbar.appendChild(
                    splitButton
                );

                toolbar.appendChild(
                    saveButton
                );

                toolbar.appendChild(
                    revertButton
                );


                wrapper.appendChild(toolbar);
                wrapper.appendChild(tableContainer);

                el.appendChild(wrapper);


                // Initial render
                loadSavedHTML();
            }
        };
        """

    return (TableStructureEditor,)


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class EditableHTMLTable(anywidget.AnyWidget):
        html = traitlets.Unicode("").tag(sync=True)
        value = traitlets.Unicode("").tag(sync=True)

        _esm = """
        export default {
            render({ model, el }) {
                el.innerHTML = "";

                const wrapper = document.createElement("div");

                Object.assign(wrapper.style, {
                    width: "100%",
                    fontFamily: "system-ui, sans-serif"
                });

                // --------------------------------------------------
                // Toolbar
                // --------------------------------------------------

                const toolbar = document.createElement("div");

                Object.assign(toolbar.style, {
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    marginBottom: "14px"
                });

                // Save button
                const saveButton = document.createElement("button");
                saveButton.textContent = "✓  Save changes";

                Object.assign(saveButton.style, {
                    background: "#2563eb",
                    color: "white",
                    border: "none",
                    borderRadius: "6px",
                    padding: "8px 14px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer"
                });

                // Revert button
                const revertButton = document.createElement("button");
                revertButton.textContent = "↶  Revert";

                Object.assign(revertButton.style, {
                    background: "white",
                    color: "#374151",
                    border: "1px solid #9ca3af",
                    borderRadius: "6px",
                    padding: "8px 14px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer"
                });

                // --------------------------------------------------
                // Table container
                // --------------------------------------------------

                const tableContainer = document.createElement("div");

                Object.assign(tableContainer.style, {
                    width: "100%",
                    overflowX: "auto"
                });

                // --------------------------------------------------
                // Build selected table
                // --------------------------------------------------

                function buildTable(html) {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, "text/html");
                    const table = doc.querySelector("table");

                    if (!table) {
                        tableContainer.textContent = "No table found.";
                        return null;
        }

                     // Add borders
                    table.style.borderCollapse = "collapse";
                    table.querySelectorAll("td, th").forEach(cell => {
                        cell.style.border = "1px solid #444";
                        cell.style.padding = "4px 8px";
                    });

                    // Make cells editable
                    table
                        .querySelectorAll("td, th")
                        .forEach((cell) => {

                            cell.setAttribute(
                                "contenteditable",
                                "true"
                            );

                            cell.style.cursor = "text";

                            cell.addEventListener(
                                "focus",
                                () => {
                                    cell.style.outline =
                                        "2px solid #2563eb";
                                    cell.style.outlineOffset =
                                        "-2px";
                                }
                            );

                            cell.addEventListener(
                                "blur",
                                () => {
                                    cell.style.outline = "";
                                }
                            );
                        });

                    tableContainer.innerHTML = "";
                    tableContainer.appendChild(table);

                    return table;
                }

                // --------------------------------------------------
                // Get current selected table
                // --------------------------------------------------

                function getCurrentTableHTML() {
                    const table =
                        tableContainer.querySelector("table");

                    return table
                        ? table.outerHTML
                        : "";
                }

                // --------------------------------------------------
                // Load table
                // --------------------------------------------------

                function loadSavedHTML() {
                    buildTable(
                        model.get("value") ||
                        model.get("html")
                    );
                }

                // Initial render
                loadSavedHTML();

                // --------------------------------------------------
                // SAVE
                // --------------------------------------------------

                saveButton.addEventListener(
                    "click",
                    () => {

                        const currentHTML =
                            getCurrentTableHTML();

                        if (!currentHTML) {
                            return;
                        }

                        model.set(
                            "value",
                            currentHTML
                        );

                        model.save_changes();

                        saveButton.textContent =
                            "✓  Saved!";

                        setTimeout(
                            () => {
                                saveButton.textContent =
                                    "✓  Save changes";
                            },
                            1000
                        );
                    }
                );

                // --------------------------------------------------
                // REVERT
                // --------------------------------------------------

                revertButton.addEventListener(
                    "click",
                    () => {
                        loadSavedHTML();
                    }
                );

                // --------------------------------------------------
                // Assemble
                // --------------------------------------------------

                toolbar.appendChild(saveButton);
                toolbar.appendChild(revertButton);

                wrapper.appendChild(toolbar);
                wrapper.appendChild(tableContainer);

                el.appendChild(wrapper);
            }
        };
        """

    return (EditableHTMLTable,)


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class ImageCropWidget(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
          const overlay = document.createElement("div");
          overlay.style.position = "relative";
          overlay.style.display = "inline-block";

          const img = document.createElement("img");
          img.src = model.get("image_src");
          img.style.maxWidth = "100%";
          img.style.cursor = "crosshair";
          img.draggable = false;
          overlay.appendChild(img);

          // keep the displayed image in sync if image_src changes
          model.on("change:image_src", () => {
            img.src = model.get("image_src");
          });

          let box = null;
          let dragging = false;
          let startX = 0, startY = 0;

          img.addEventListener("mousedown", (e) => {
            dragging = true;
            startX = e.offsetX;
            startY = e.offsetY;
            if (box) box.remove();
            box = document.createElement("div");
            box.style.position = "absolute";
            box.style.border = "2px solid red";
            box.style.pointerEvents = "none";   // so offsetX stays relative to <img>
            box.style.left = startX + "px";
            box.style.top = startY + "px";
            overlay.appendChild(box);
            e.preventDefault();
          });

          window.addEventListener("mousemove", (e) => {
            if (!dragging || !box) return;
            const r = img.getBoundingClientRect();
            const cx = Math.max(0, Math.min(e.clientX - r.left, r.width));
            const cy = Math.max(0, Math.min(e.clientY - r.top, r.height));
            box.style.left = Math.min(startX, cx) + "px";
            box.style.top = Math.min(startY, cy) + "px";
            box.style.width = Math.abs(cx - startX) + "px";
            box.style.height = Math.abs(cy - startY) + "px";
          });

          window.addEventListener("mouseup", (e) => {
            if (!dragging) return;
            dragging = false;
            const r = img.getBoundingClientRect();
            const cx = Math.max(0, Math.min(e.clientX - r.left, r.width));
            const cy = Math.max(0, Math.min(e.clientY - r.top, r.height));
            // convert display px -> natural image px (img may be scaled by maxWidth)
            const sx = img.naturalWidth / r.width;
            const sy = img.naturalHeight / r.height;
            model.set("crop", {
              x: Math.round(Math.min(startX, cx) * sx),
              y: Math.round(Math.min(startY, cy) * sy),
              width: Math.round(Math.abs(cx - startX) * sx),
              height: Math.round(Math.abs(cy - startY) * sy),
            });
            model.save_changes();   // sync back to Python only once, on release
          });

          el.appendChild(overlay);
        }
        export default { render };
        """

        image_src = traitlets.Unicode("").tag(sync=True)
        crop = traitlets.Dict({}).tag(sync=True)


    return (ImageCropWidget,)


if __name__ == "__main__":
    app.run()