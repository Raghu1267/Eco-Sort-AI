"""ECO SORT AI - AI-Based Smart E-Waste Segregation and Classification System.

Run with:  streamlit run app.py
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from modules import charts, ui
from modules.detection import (
    DETECTION_COLUMNS,
    ImageValidationError,
    ModelLoadError,
    get_class_names,
    image_to_png_bytes,
    load_model,
    run_detection,
    validate_image,
)
from modules.segregation import (
    DISCLAIMER,
    GENERAL_HAZARD_WARNING,
    HAZARD_ORDER,
    SEGREGATION_RULES,
    add_segregation,
    check_mapping,
    count_by,
)

APP_DIR = Path(__file__).resolve().parent
MAX_HISTORY_RUNS = 25
HAZARD_DOT = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}

st.set_page_config(
    page_title="ECO SORT AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the trained YOLOv8s model (first run only)...")
def get_model():
    """Load best.pt once per server process and reuse it for every request."""
    return load_model()


def init_state() -> None:
    st.session_state.setdefault("last", None)  # most recent detection run
    st.session_state.setdefault("history", [])  # DataFrames of all runs this session
    st.session_state.setdefault("run_counter", 0)


def store_run(filename: str, result) -> dict:
    """Save a detection run (with segregation columns) to the session."""
    st.session_state["run_counter"] += 1
    run_no = st.session_state["run_counter"]
    full = add_segregation(result.detections)
    full.insert(0, "Image", filename)
    full.insert(0, "Run", run_no)
    run = {
        "run": run_no,
        "name": filename,
        "time": datetime.now().strftime("%H:%M:%S"),
        "result": result,
        "png": image_to_png_bytes(result.annotated_image),
        "df": full,
    }
    st.session_state["last"] = run
    history = st.session_state["history"]
    history.append(full)
    del history[:-MAX_HISTORY_RUNS]
    return run


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # BOM so Excel reads UTF-8 correctly


def detection_report(run: dict) -> pd.DataFrame:
    res = run["result"]
    df = res.detections[DETECTION_COLUMNS].copy()
    df.insert(0, "Image", run["name"])
    df["Confidence Threshold (%)"] = round(res.conf_threshold * 100, 1)
    return df


def empty_state(message: str) -> None:
    ui.note(message)
    st.write("")
    st.page_link(PAGE_DETECT, label="Go to AI Detection", icon=":material/search:")


# --------------------------------------------------------------------------- #
# Page 1: Home
# --------------------------------------------------------------------------- #
def page_home() -> None:
    ui.hero()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.stat_card(str(len(SEGREGATION_RULES)), "Supported e-waste classes")
    with c2:
        ui.stat_card("YOLOv8s", "AI model")
    with c3:
        ui.stat_card("7", "E-waste categories")
    with c4:
        ui.stat_card("CPU", "Inference, no GPU needed")

    st.write("")
    if st.button("Upload an image", type="primary", icon=":material/upload:"):
        st.switch_page(PAGE_DETECT)

    st.write("")
    st.subheader("How the system works")
    left, right = st.columns(2, gap="large")
    with left:
        ui.step_card("1. Upload", "Add a JPG, JPEG or PNG photo that shows one or more electronic items.")
        ui.step_card("2. Detect", "The trained YOLOv8s model finds each item and predicts its class and confidence.")
    with right:
        ui.step_card("3. Segregate", "A rule-based engine maps each detected class to a category, bin and hazard level.")
        ui.step_card("4. Act and export", "Read the disposal and safety guidance, then download the annotated image or CSV report.")

    st.write("")
    st.subheader("What you get")
    features = [
        ("🎯", "Multi-object detection", "Finds several e-waste items in one image and draws a box around each."),
        ("🧭", "Smart segregation", "Category, collection stream and hazard priority for every detected item."),
        ("⚠️", "Safety first", "Warnings for batteries, fluorescent lamps, smoke detectors and other hazardous items."),
        ("📊", "Session analytics", "Charts built only from the images you process in this session."),
        ("🎚️", "Adjustable threshold", "Tune the confidence cut-off to balance missed items against false alarms."),
        ("📥", "Downloadable reports", "Save the annotated image and a CSV report of every detection."),
    ]
    for row in (features[:3], features[3:]):
        cols = st.columns(3, gap="medium")
        for col, (icon, title, body) in zip(cols, row):
            with col:
                ui.info_card(icon, title, body)
        st.write("")

    st.subheader("Why e-waste segregation matters")
    a, b, c = st.columns(3, gap="medium")
    with a:
        ui.info_card("🧪", "Hazardous substances", "Batteries, lamps and old displays can contain lead, mercury or cadmium that harm soil and water if dumped.")
    with b:
        ui.info_card("🔁", "Valuable materials", "Circuit boards and devices hold copper, gold, silver and other metals that recycling can recover.")
    with c:
        ui.info_card("🔥", "Fire and health risk", "Crushed lithium batteries and informal burning release toxic fumes and start fires at waste sites.")
    st.write("")
    ui.note(
        "According to the Global E-waste Monitor 2024, the world generated about 62 million tonnes of e-waste in 2022, "
        "and only around 22% was documented as formally collected and recycled. Check the latest edition for current figures."
    )


# --------------------------------------------------------------------------- #
# Page 2: AI Detection
# --------------------------------------------------------------------------- #
def render_detection_results(run: dict) -> None:
    res = run["result"]
    df = res.detections
    st.divider()
    st.subheader("Detection results")
    st.caption(f"Image: {run['name']}  |  confidence threshold {res.conf_threshold:.0%}  |  run #{run['run']} at {run['time']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Objects detected", len(df))
    m2.metric("Unique classes", df["Class"].nunique() if len(df) else 0)
    m3.metric("Average confidence", f"{df['Confidence (%)'].mean():.1f}%" if len(df) else "n/a")
    m4.metric("Inference time", f"{res.inference_ms:.0f} ms" if res.inference_ms else f"{res.total_ms:.0f} ms")

    st.write("")
    if df.empty:
        st.info(
            f"No e-waste objects were detected at a confidence of {res.conf_threshold:.0%} or higher. "
            "Try lowering the threshold, or use a sharper, well-lit photo where the item fills more of the frame."
        )
        st.image(res.annotated_image, caption="Uploaded image (no detections)")
        return

    st.image(res.annotated_image, caption="Annotated image: class name and confidence on every box")
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "Confidence (%)": st.column_config.ProgressColumn(
                "Confidence (%)", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )

    stem = Path(run["name"]).stem
    d1, d2, d3 = st.columns([1, 1, 2])
    with d1:
        st.download_button(
            "Download annotated image", data=run["png"], file_name=f"{stem}_annotated.png",
            mime="image/png", type="primary", icon=":material/image:",
        )
    with d2:
        st.download_button(
            "Download CSV report", data=csv_bytes(detection_report(run)), file_name=f"{stem}_detections.csv",
            mime="text/csv", icon=":material/download:",
        )
    with d3:
        st.page_link(PAGE_SEG, label="View segregation guidance for these items", icon=":material/recycling:")


def page_detection() -> None:
    ui.page_header("AI Detection", "Upload an image, set the confidence threshold and run the trained YOLOv8s model.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        uploaded = st.file_uploader(
            "Upload an e-waste image",
            type=["jpg", "jpeg", "png"],
            help="JPG, JPEG or PNG, up to 15 MB. Photos with several items are supported.",
        )
    with right:
        conf = st.slider(
            "Confidence threshold", 0.05, 0.95, st.session_state.get("conf_saved", 0.25), 0.05,
            help="Detections below this confidence are hidden. Lower values find more items but add false alarms.",
        )
        st.session_state["conf_saved"] = conf
        with st.expander("Advanced settings"):
            iou = st.slider(
                "NMS IoU threshold", 0.10, 0.90, st.session_state.get("iou_saved", 0.45), 0.05,
                help="Controls how overlapping boxes are merged. Lower values remove more overlapping boxes.",
            )
            st.session_state["iou_saved"] = iou
        detect = st.button("Detect objects", type="primary", disabled=uploaded is None, icon=":material/search:")
        if uploaded is None:
            st.caption("Upload an image to enable detection.")

    image = None
    if uploaded is not None:
        try:
            image = validate_image(uploaded.getvalue(), uploaded.name)
        except ImageValidationError as exc:
            st.error(str(exc))
        else:
            with left:
                st.image(image, caption=f"Preview: {uploaded.name} ({image.width} x {image.height} px)")

    if detect and image is not None:
        try:
            model = get_model()
        except ModelLoadError as exc:
            st.error(f"The model could not be loaded. {exc}")
        else:
            problems = check_mapping(get_class_names(model).values())
            if problems["unmapped"]:
                st.warning(
                    "Some model classes have no segregation rule and will show a generic fallback: "
                    + ", ".join(problems["unmapped"])
                )
            try:
                with st.spinner("Running YOLOv8s inference on CPU..."):
                    result = run_detection(model, image, conf=conf, iou=iou)
            except Exception as exc:  # surface any runtime failure instead of crashing the page
                st.error(f"Detection failed: {exc}")
            else:
                store_run(uploaded.name, result)

    run = st.session_state.get("last")
    if run:
        if uploaded is not None and image is not None and uploaded.name != run["name"]:
            st.info("A new image is selected. Click Detect objects to update the results below.")
        render_detection_results(run)


# --------------------------------------------------------------------------- #
# Page 3: Smart Segregation
# --------------------------------------------------------------------------- #
def render_item_card(row: pd.Series) -> None:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.render_html(
            ui.field("E-waste category", escape(row["E-Waste Category"]))
            + ui.field("Segregation bin", f"<b>{escape(row['Segregation Bin'])}</b>")
            + ui.field("Hazard priority", ui.hazard_badge(row["Hazard Priority"]))
        )
    with c2:
        ui.render_html(
            ui.field("Disposal guidance", escape(row["Disposal Guidance"]))
            + ui.field("Recycling recommendation", escape(row["Recycling Recommendation"]))
        )
    if row["Hazard Priority"] == "High":
        ui.warning_box("Safety instructions", [row["Safety Instructions"]])
    else:
        ui.render_html(ui.field("Safety instructions", escape(row["Safety Instructions"])))


def page_segregation() -> None:
    ui.page_header("Smart Segregation", "Rule-based category, collection stream, hazard level and handling guidance for each detected item.")
    run = st.session_state.get("last")
    if not run:
        empty_state("No detections yet. Run a detection first and the segregation guidance will appear here.")
        return
    df = run["df"]
    st.caption(f"Image: {run['name']}  |  run #{run['run']} at {run['time']}")
    if df.empty:
        st.info("No objects were detected in the last image, so there is nothing to segregate. Lower the confidence threshold and try again.")
        return

    ui.note(DISCLAIMER)
    st.write("")

    high = df[df["Hazard Priority"] == "High"]
    if not high.empty:
        ui.warning_box(
            f"{len(high)} high-hazard item(s) detected: " + ", ".join(sorted(high["Class"].unique())),
            [GENERAL_HAZARD_WARNING, "Keep these items separate from the other e-waste and hand them to an authorised collector."],
        )
        st.write("")

    st.subheader("Segregation summary")
    bins = count_by(df, "Segregation Bin")
    cols_per_row = 4
    records = list(bins.itertuples(index=False))
    for start in range(0, len(records), cols_per_row):
        cols = st.columns(cols_per_row, gap="small")
        for col, rec in zip(cols, records[start:start + cols_per_row]):
            with col:
                ui.bin_card(int(rec.Count), rec[0])
        st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.bar_counts(df, "Segregation Bin", "Objects per segregation bin"), key="chart_1")
    with c2:
        st.plotly_chart(charts.bar_counts(df, "E-Waste Category", "Objects per e-waste category"), key="chart_2")

    st.subheader("Guidance for each detected object")
    rank = {h: i for i, h in enumerate(HAZARD_ORDER)}
    ordered = df.assign(_r=df["Hazard Priority"].map(rank)).sort_values(["_r", "Detection ID"])
    for _, row in ordered.iterrows():
        label = f"{HAZARD_DOT.get(row['Hazard Priority'], '')} #{row['Detection ID']}  {row['Class']}  ({row['Confidence (%)']:.1f}%)  {row['Hazard Priority']} hazard"
        with st.expander(label, expanded=row["Hazard Priority"] == "High"):
            render_item_card(row)

    st.write("")
    drop = ["Run"]
    st.download_button(
        "Download segregation report (CSV)",
        data=csv_bytes(df.drop(columns=drop)),
        file_name=f"{Path(run['name']).stem}_segregation.csv",
        mime="text/csv", type="primary", icon=":material/download:",
    )


# --------------------------------------------------------------------------- #
# Page 4: Analytics
# --------------------------------------------------------------------------- #
def page_analytics() -> None:
    ui.page_header("Analytics Dashboard", "Charts built from the detections you ran in this session. Nothing is simulated or stored between sessions.")
    history = st.session_state.get("history", [])
    if not history:
        empty_state("No detections yet. Process an image and the analytics will appear here.")
        return

    scope = "Latest image"
    if len(history) > 1:
        scope = st.radio("Data shown", ["Latest image", "All images this session"], horizontal=True)
    df = history[-1] if scope == "Latest image" else pd.concat(history, ignore_index=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total objects detected", len(df))
    m2.metric("Unique classes", df["Class"].nunique() if len(df) else 0)
    m3.metric("Average confidence", f"{df['Confidence (%)'].mean():.1f}%" if len(df) else "n/a")
    m4.metric("Images processed", len(history) if scope != "Latest image" else 1)

    if df.empty:
        st.info("No objects were detected in the selected data. Lower the confidence threshold and run again.")
    st.write("")

    r1a, r1b = st.columns(2)
    with r1a:
        st.plotly_chart(charts.bar_counts(df, "Class", "Category-wise distribution (detected classes)"), key="chart_3")
    with r1b:
        st.plotly_chart(charts.confidence_histogram(df), key="chart_4")
    r2a, r2b = st.columns(2)
    with r2a:
        st.plotly_chart(charts.hazard_donut(df), key="chart_5")
    with r2b:
        st.plotly_chart(charts.bar_counts(df, "Segregation Bin", "Segregation category distribution"), key="chart_6")

    st.write("")
    if st.button("Clear session data", icon=":material/delete:"):
        st.session_state["last"] = None
        st.session_state["history"] = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Page 5: About
# --------------------------------------------------------------------------- #
def page_about() -> None:
    ui.page_header("About the Project", "AI-Based Smart E-Waste Segregation and Classification System Using YOLOv8")

    st.subheader("The e-waste problem")
    st.write(
        "Discarded electronics are one of the fastest-growing waste streams. They hold valuable metals but also "
        "hazardous substances, and mixing them with ordinary waste wastes the first and spreads the second. "
        "Sorting items correctly at the start is the cheapest way to make safe recycling possible, yet it is slow and "
        "error-prone when done by eye."
    )

    st.subheader("Project objectives")
    st.markdown(
        "- Detect common e-waste items in a photo with a trained object detector.\n"
        "- Report the class and confidence of every detected item.\n"
        "- Map each item to an e-waste category, collection stream and hazard priority.\n"
        "- Give disposal, recycling and safety guidance, with clear warnings for hazardous items.\n"
        "- Run on a normal CPU so it can be deployed for free on Streamlit Community Cloud."
    )

    st.subheader("YOLOv8 object detection")
    st.write(
        "YOLOv8 is a single-stage object detector: one forward pass over the image predicts boxes, class labels and "
        "confidence scores for all objects at once. This project uses the small variant, YOLOv8s, fine-tuned on "
        f"{len(SEGREGATION_RULES)} e-waste classes."
    )

    st.subheader("How the detection pipeline works")
    st.markdown(
        "1. The uploaded file is validated (type, size, readable JPEG or PNG) and corrected for phone rotation.\n"
        "2. The image is passed to the trained model (`model/best.pt`) on CPU.\n"
        "3. The model returns boxes, class ids and confidence scores. Boxes below your threshold are dropped, and "
        "overlapping boxes are merged by non-maximum suppression.\n"
        "4. Class names are read from the model itself, and boxes are drawn on the image.\n"
        "5. Results are shown in tables, exported as CSV and passed to the segregation engine."
    )

    st.subheader("How rule-based segregation works")
    st.write(
        "The segregation engine is a lookup table, not a second AI model. Each of the 37 class names is linked to a "
        "category, a collection stream, a hazard priority (High, Medium or Low) and written disposal, recycling and "
        "safety guidance. Hazard priority is set by what that kind of item usually contains, for example refrigerant, "
        "mercury or lithium batteries. It does not inspect the actual item."
    )
    ui.note(DISCLAIMER)
    st.write("")

    st.subheader("Model performance")
    metrics = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-score", "mAP@50", "mAP@50-95"],
            "Value": ["92.89%", "86.33%", "89.49%", "92.04%", "75.94%"],
            "Meaning": [
                "Share of predicted boxes that are correct",
                "Share of real objects that were found",
                "Balance of precision and recall",
                "Mean average precision at IoU 0.50",
                "Mean average precision averaged over IoU 0.50 to 0.95",
            ],
        }
    )
    st.dataframe(metrics, hide_index=True)
    st.caption(
        "These are results from the test-set evaluation of the trained YOLOv8s model. mAP is an object-detection "
        "metric that scores both the class and the box position. It is not classification accuracy."
    )

    st.subheader("Technology stack")
    st.markdown(
        "- **Model:** Ultralytics YOLOv8s, PyTorch (CPU inference)\n"
        "- **Interface:** Streamlit\n"
        "- **Image handling:** Pillow, OpenCV, NumPy\n"
        "- **Data and charts:** Pandas, Plotly\n"
        "- **Deployment:** GitHub and Streamlit Community Cloud"
    )

    st.subheader("Limitations")
    st.markdown(
        "- The model knows only the 37 trained classes. Other objects may be missed or labelled as the closest class.\n"
        "- Accuracy drops with poor lighting, blur, heavy overlap, tiny objects and unusual angles.\n"
        "- Guidance depends on the predicted class only. The app cannot tell whether a device is damaged, contains a "
        "specific battery type or holds personal data.\n"
        "- Recommendations are general and are not legally certified waste classifications. Local rules always apply.\n"
        "- Inference on a free CPU host can take a few seconds per image."
    )

    st.subheader("Future scope")
    st.markdown(
        "- Add more classes and more training images, especially for under-represented items.\n"
        "- Region-specific rule sets that follow national e-waste regulations.\n"
        "- Detect battery or hazardous components inside a device.\n"
        "- Batch upload and image-quality checks.\n"
        "- Model export (ONNX or quantised) for faster CPU or mobile inference."
    )


# --------------------------------------------------------------------------- #
# Navigation, sidebar and entry point
# --------------------------------------------------------------------------- #
PAGE_HOME = st.Page(page_home, title="Home", icon=":material/home:", url_path="home", default=True)
PAGE_DETECT = st.Page(page_detection, title="AI Detection", icon=":material/search:", url_path="detection")
PAGE_SEG = st.Page(page_segregation, title="Smart Segregation", icon=":material/recycling:", url_path="segregation")
PAGE_ANALYTICS = st.Page(page_analytics, title="Analytics", icon=":material/monitoring:", url_path="analytics")
PAGE_ABOUT = st.Page(page_about, title="About", icon=":material/info:", url_path="about")


def render_sidebar() -> None:
    logo = APP_DIR / "assets" / "logo.svg"
    if logo.exists():
        try:
            st.logo(str(logo))
        except Exception:
            pass  # older Streamlit versions: skip the logo
    with st.sidebar:
        st.divider()
        history = st.session_state.get("history", [])
        total = int(sum(len(h) for h in history))
        st.caption(f"Session: {len(history)} image(s) processed, {total} object(s) detected")
        st.caption("Recommendations are general guidance, not certified waste classifications.")


init_state()
ui.inject_css()
navigation = st.navigation([PAGE_HOME, PAGE_DETECT, PAGE_SEG, PAGE_ANALYTICS, PAGE_ABOUT])
render_sidebar()
navigation.run()
