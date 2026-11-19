"""Shop Insights: upload your sales file and understand your shop in minutes.

Landing page -> upload (or demo) -> dashboard with a weekly to-do list, plain-English
insights, sales charts, customer segments (RFM), a churn-risk win-back list with a
ready message, and a downloadable Excel report. Data is processed in memory only.
"""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

_LOGO_FILE = Path(__file__).with_name("logo.png")
st.set_page_config(page_title="Shop Insights", page_icon=str(_LOGO_FILE) if _LOGO_FILE.exists() else "📊",
                   layout="wide",
                   initial_sidebar_state="collapsed")

INK, TEAL, AMBER, CORAL, LEAF = "#17211F", "#0F4C4A", "#F2A93B", "#E26D5C", "#3FA796"
SEG_COLORS = {"Champions": TEAL, "Loyal": LEAF, "At Risk": AMBER, "Lost": CORAL}
REQUIRED = ("date", "customer", "order", "price")
LOGO_SVG = ('<svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" aria-label="Shop Insights logo">'
            '<rect width="40" height="40" rx="12" fill="#0F4C4A"/><path d="M8 15 10.5 9H29.5L32 15Z" fill="#FFFFFF"/>'
            '<path d="M8 15a4 4 0 0 0 8 0a4 4 0 0 0 8 0a4 4 0 0 0 8 0Z" fill="#F2A93B"/>'
            '<rect x="12" y="27" width="4" height="5" rx="1" fill="#FFFFFF"/><rect x="18" y="24" width="4" height="8" rx="1" fill="#FFFFFF"/>'
            '<rect x="24" y="21" width="4" height="11" rx="1" fill="#FFFFFF"/></svg>')

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=DM+Sans:wght@400;500;600&display=swap');
.stApp, .stApp div, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input, .stApp textarea { font-family: 'DM Sans', sans-serif; }
.stApp { background: #F4F7F6; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Bricolage Grotesque', sans-serif; color: #17211F; letter-spacing: -0.01em; }
#MainMenu, footer, .stAppDeployButton { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container { padding-top: 1.2rem; max-width: 1180px; }

.stApp .brand { display: flex; align-items: center; gap: 10px; font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.35rem; color: #17211F; }
.stApp .logo { width: 38px; height: 38px; border-radius: 12px; background: #0F4C4A; display: flex; align-items: center; justify-content: center; font-size: 1.15rem; }

.stApp .hero { background: #0F4C4A; border-radius: 24px; padding: 38px 44px; margin: 10px 0 22px 0; display: flex; align-items: center; justify-content: space-between; gap: 36px; }
.stApp .hero-main { flex: 1; min-width: 0; }
.stApp .hero-tag { display: inline-block; background: rgba(242,169,59,0.18); color: #F7C877; font-weight: 600; font-size: 0.85rem; padding: 4px 12px; border-radius: 999px; margin-bottom: 14px; }
.stApp .hero-title { font-family: 'Bricolage Grotesque', sans-serif; color: #FFFFFF; font-weight: 700; font-size: 2.4rem; line-height: 1.12; letter-spacing: -0.02em; max-width: 720px; }
.stApp .hero-sub { color: #BFE0DA; margin-top: 14px; font-size: 1.02rem; max-width: 640px; line-height: 1.5; }
.stApp .hero-art { flex: none; text-align: right; }
.stApp .hero-cap { color: #9FD3C9; font-size: 0.82rem; margin-top: 4px; }

.stApp .kpi { background: #FFFFFF; border: 1px solid #DDE6E3; border-radius: 16px; padding: 18px 20px; height: 100%; }
.stApp .kpi.strong { background: #E3F1ED; border-color: #C5E0D9; }
.stApp .kpi-label { color: #5B6B67; font-size: 0.88rem; font-weight: 500; }
.stApp .kpi-value { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.85rem; font-weight: 700; color: #17211F; margin-top: 4px; }
.stApp .kpi-note { color: #5B6B67; font-size: 0.82rem; margin-top: 2px; }
.stApp .up { color: #1E8E5A; font-weight: 600; }
.stApp .down { color: #C4493A; font-weight: 600; }

.stApp .insight { display: flex; gap: 14px; align-items: flex-start; padding: 13px 2px; border-bottom: 1px solid #E6EDEA; color: #17211F; }
.stApp .insight:last-child { border-bottom: none; }
.stApp .ico { width: 36px; height: 36px; flex: none; border-radius: 10px; background: #E3F1ED; display: flex; align-items: center; justify-content: center; font-size: 1.05rem; }

.stApp .seg { background: #FFFFFF; border: 1px solid #DDE6E3; border-radius: 16px; padding: 16px 18px; height: 100%; }
.stApp .seg-name { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.15rem; color: #17211F; }
.stApp .seg-stat { color: #17211F; font-weight: 600; margin-top: 6px; }
.stApp .seg-do { color: #5B6B67; font-size: 0.92rem; margin-top: 6px; line-height: 1.45; }
.stApp .muted { color: #5B6B67; }

[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-color: #DDE6E3; border-radius: 16px; }
[data-testid="stFileUploaderDropzone"] { border: 2px dashed #9FC9C0; border-radius: 14px; background: #F4FAF8; }
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button { border-radius: 12px; font-weight: 600; }
.stDownloadButton button:hover { border-color: #0F4C4A; color: #0F4C4A; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: none; }
.stTabs [data-baseweb="tab"] { height: 42px; padding: 0 20px; border-radius: 999px; background: #FFFFFF; border: 1px solid #D5E0DC; }
.stTabs [data-baseweb="tab"] p { font-weight: 600; color: #3B4A47; }
.stTabs [aria-selected="true"] { background: #0F4C4A; border-color: #0F4C4A; }
.stTabs [aria-selected="true"] p { color: #FFFFFF; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

@media (max-width: 900px) { .stApp .hero-art { display: none; } }
@media (max-width: 640px) { .stApp .hero { padding: 26px 22px; } .stApp .hero-title { font-size: 1.65rem; } }
</style>
"""

ALIASES = {
    "date": ["invoicedate", "date", "orderdate", "purchasedate", "timestamp", "datetime", "billdate"],
    "customer": ["customerid", "customer", "clientid", "userid", "customername", "email", "phone", "mobile"],
    "order": ["invoiceno", "invoice", "orderid", "orderno", "order", "billno", "bill", "transactionid"],
    "product": ["description", "product", "item", "productname", "itemname", "sku", "stockcode"],
    "quantity": ["quantity", "qty", "units"],
    "price": ["unitprice", "price", "rate", "mrp", "saleprice", "amount"],
    "place": ["country", "city", "region", "state", "location", "area"],
}


# ---------------------------------------------------------------- data helpers
def guess(cols, key):
    norm = {c: str(c).lower().replace(" ", "").replace("_", "") for c in cols}
    for alias in ALIASES[key]:
        for c, n in norm.items():
            if n == alias:
                return c
    return None


@st.cache_data
def demo_data():
    """Synthetic demo shop (clearly labelled in the UI)."""
    rng = np.random.default_rng(42)
    items = {
        "Masala Tea 250g": 120, "Filter Coffee 500g": 340, "Basmati Rice 5kg": 560,
        "Sunflower Oil 1L": 150, "Toor Dal 1kg": 160, "Notebook A4": 60, "Ball Pen Pack": 40,
        "Soap Bar": 35, "Shampoo 340ml": 210, "Biscuit Family Pack": 55, "Ghee 500ml": 330,
        "Detergent 1kg": 120, "Snack Mix": 90, "Cooking Spice Box": 450, "Water Bottle 1L": 99,
    }
    names = list(items)
    w = 1 / np.arange(1, len(names) + 1) ** 0.7
    w = w / w.sum()
    places = ["Chennai", "Tambaram", "Velachery", "Anna Nagar", "Porur"]
    start, end = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-10-31")
    rows, oid = [], 1000
    for cid in range(1, 1201):
        first = start + pd.Timedelta(days=int(rng.integers(0, 600)))
        last = min(first + pd.Timedelta(days=int(rng.exponential(250)) + 30), end)
        gap = int(rng.integers(15, 70))
        place = rng.choice(places)
        t = first
        while t <= last:
            oid += 1
            when = t + pd.Timedelta(hours=int(rng.integers(9, 21)), minutes=int(rng.integers(0, 60)))
            for _ in range(int(rng.integers(1, 4))):
                p = rng.choice(names, p=w)
                rows.append((f"C{cid:04d}", f"O{oid}", when, p, int(rng.integers(1, 4)), items[p], place))
            t += pd.Timedelta(days=int(rng.integers(max(gap // 2, 1), gap * 2)))
    return pd.DataFrame(rows, columns=["CustomerID", "OrderID", "OrderDate", "Product",
                                       "Quantity", "UnitPrice", "City"])


def read_upload(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    raw = file.getvalue()
    for enc in ("utf-8", "ISO-8859-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8", errors="ignore")


def prepare(raw, m):
    """Map the user's columns to a standard table and clean it."""
    d = pd.DataFrame(index=raw.index)
    d["Date"] = pd.to_datetime(raw[m["date"]], errors="coerce")
    d["Customer"] = raw[m["customer"]]
    d["Order"] = raw[m["order"]].astype(str)
    d["Product"] = raw[m["product"]].astype(str) if m["product"] else "All products"
    d["Quantity"] = pd.to_numeric(raw[m["quantity"]], errors="coerce") if m["quantity"] else 1
    price = pd.to_numeric(raw[m["price"]], errors="coerce")
    d["Revenue"] = d["Quantity"] * price
    d["Place"] = raw[m["place"]].astype(str) if m["place"] else "All"
    before = len(d)
    d = d.dropna(subset=["Date", "Customer", "Revenue", "Quantity"])
    d = d[(d["Revenue"] > 0) & (d["Quantity"] > 0)]  # drops returns / bad rows
    d["Customer"] = d["Customer"].astype(str)
    return d.reset_index(drop=True), before - len(d)


def features(d, snapshot):
    g = d.groupby("Customer")
    f = pd.DataFrame({
        "Recency": (snapshot - g["Date"].max()).dt.days,
        "Frequency": g["Order"].nunique(),
        "Monetary": g["Revenue"].sum(),
        "Tenure": (snapshot - g["Date"].min()).dt.days,
        "Items": g["Quantity"].sum(),
    })
    f["AvgOrderValue"] = f["Monetary"] / f["Frequency"]
    return f


@st.cache_data
def rfm_segments(d):
    snapshot = d["Date"].max() + pd.Timedelta(days=1)
    f = features(d, snapshot)
    if len(f) < 8:
        return None
    f["R"] = pd.qcut(f["Recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    f["F"] = pd.qcut(f["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    f["M"] = pd.qcut(f["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    score = f["R"] + f["F"] + f["M"]
    f["Segment"] = pd.cut(score, [0, 4, 7, 9, 12], labels=["Lost", "At Risk", "Loyal", "Champions"])
    return f.reset_index()


@st.cache_data
def churn_scores(d, window):
    """Return (metrics or None, importance or None, scored customers)."""
    snapshot = d["Date"].max() + pd.Timedelta(days=1)
    now = features(d, snapshot)
    cutoff = d["Date"].max() - pd.Timedelta(days=window)
    past = d[d["Date"] <= cutoff]
    future_buyers = set(d[d["Date"] > cutoff]["Customer"])

    metrics = importance = None
    if past["Customer"].nunique() >= 100:
        X = features(past, cutoff)
        y = (~X.index.isin(future_buyers)).astype(int)
        if y.sum() >= 10 and (len(y) - y.sum()) >= 10:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            model = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                           random_state=42, n_jobs=-1).fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            pred = model.predict(X_te)
            metrics = {"Recall": recall_score(y_te, pred), "Precision": precision_score(y_te, pred),
                       "ROC-AUC": roc_auc_score(y_te, proba), "Churn rate": float(y.mean()),
                       "Customers": len(X)}
            importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
            now["ChurnRisk"] = model.predict_proba(now[X.columns])[:, 1]
    if metrics is None:  # not enough data for ML: simple rule
        now["ChurnRisk"] = (now["Recency"] / (window * 2)).clip(0, 1)
    return metrics, importance, now.reset_index()


# ------------------------------------------------------------ display helpers
def fmt(n, money):
    """Indian digit grouping (12,34,567) when the currency is the rupee."""
    n = int(round(n))
    s = str(abs(n))
    if money == "₹" and len(s) > 3:
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]
    else:
        s = f"{abs(n):,}"
    return ("-" if n < 0 else "") + s


def bold(text):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def kpi(label, value, note="", strong=False):
    cls = "kpi strong" if strong else "kpi"
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    return (f'<div class="{cls}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{note_html}</div>')


def card(title, text, color=None, icon=None):
    top = f' style="border-top:5px solid {color}"' if color else ""
    ico = f'<div class="ico" style="margin-bottom:10px">{icon}</div>' if icon else ""
    return f'<div class="seg"{top}>{ico}<div class="seg-name">{title}</div><div class="seg-do">{text}</div></div>'


def sparkline(d):
    m = d.set_index("Date").resample("MS")["Revenue"].sum()
    if len(m) < 3:
        return ""
    v = m.values.astype(float)
    lo, span = v.min(), (v.max() - v.min()) or 1.0
    W, H, pad = 320, 110, 8
    pts = [(pad + i * (W - 2 * pad) / (len(v) - 1), H - pad - (x - lo) / span * (H - 2 * pad)) for i, x in enumerate(v)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{pts[0][0]:.1f},{H - pad} {line} {pts[-1][0]:.1f},{H - pad}"
    return (f'<div class="hero-art"><svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">'
            f'<polygon points="{area}" fill="rgba(159,211,201,0.18)"/>'
            f'<polyline points="{line}" fill="none" stroke="#9FD3C9" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="5" fill="#F2A93B"/></svg>'
            f'<div class="hero-cap">Monthly revenue</div></div>')


def hero(title, sub, tag="", art=""):
    tag_html = f'<div class="hero-tag">{tag}</div>' if tag else ""
    st.markdown(f'<div class="hero"><div class="hero-main">{tag_html}<div class="hero-title">{title}</div>'
                f'<div class="hero-sub">{sub}</div></div>{art}</div>', unsafe_allow_html=True)


def style(fig, height=340, bar_color=None):
    fig.update_layout(
        template="simple_white", height=height, margin=dict(l=8, r=8, t=52, b=8),
        font=dict(family="DM Sans, sans-serif", color=INK, size=13),
        title=dict(font=dict(family="Bricolage Grotesque, sans-serif", size=18), x=0, xref="paper"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(title=None, orientation="h", y=-0.12),
    )
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="#E6EDEA", zeroline=False, title=None)
    if bar_color:
        fig.update_traces(marker_color=bar_color)
    return fig


def trend(d):
    end = d["Date"].max()
    last = d[d["Date"] > end - pd.Timedelta(days=30)]["Revenue"].sum()
    prev = d[(d["Date"] <= end - pd.Timedelta(days=30)) & (d["Date"] > end - pd.Timedelta(days=60))]["Revenue"].sum()
    return last, prev, ((last - prev) / prev if prev > 0 else None)


def insights(d, seg, money):
    out = []
    cust_rev = d.groupby("Customer")["Revenue"].sum().sort_values(ascending=False)
    top_n = max(1, int(round(len(cust_rev) * 0.2)))
    out.append(("👑", f"Your top 20% of customers ({top_n:,} people) bring **{cust_rev.head(top_n).sum() / cust_rev.sum():.0%}** "
                      f"of your revenue. Look after them first."))
    repeat = (d.groupby("Customer")["Order"].nunique() > 1).mean()
    out.append(("🔁", f"**{repeat:.0%}** of customers have bought more than once."))
    day = d.groupby(d["Date"].dt.day_name())["Revenue"].sum()
    out.append(("📅", f"**{day.idxmax()}** is your best day of the week for sales."))
    if d["Product"].nunique() > 1:
        top = d.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
        out.append(("🏆", f"Your best product is **{top.index[0]}**, worth {money}{fmt(top.iloc[0], money)} in sales."))
    _, _, ch = trend(d)
    if ch is not None:
        out.append(("📈" if ch >= 0 else "📉",
                    f"Sales in the last 30 days are **{abs(ch):.0%} {'higher' if ch >= 0 else 'lower'}** than the 30 days before."))
    if seg is not None:
        risk = seg[seg["Segment"].isin(["At Risk", "Lost"])]
        out.append(("⚠️", f"**{len(risk):,}** customers are At Risk or Lost, who together spent "
                          f"{money}{fmt(risk['Monetary'].sum(), money)}. See the Win-back tab."))
    return out


@st.cache_data
def build_report(d, seg, scored, money, notes):
    orders = d["Order"].nunique()
    revenue = d["Revenue"].sum()
    summary = pd.DataFrame({
        "Metric": ["Period", "Revenue", "Orders", "Customers", "Average order"],
        "Value": [f"{d['Date'].min():%d %b %Y} to {d['Date'].max():%d %b %Y}", f"{money}{fmt(revenue, money)}",
                  f"{orders:,}", f"{d['Customer'].nunique():,}", f"{money}{fmt(revenue / max(orders, 1), money)}"]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame({"What your data says": [re.sub(r"\*\*", "", n) for n in notes]}).to_excel(
            xw, sheet_name="Summary", index=False, startrow=len(summary) + 2)
        monthly = d.set_index("Date").resample("MS")["Revenue"].sum().round(0).reset_index()
        monthly["Date"] = monthly["Date"].dt.strftime("%b %Y")
        monthly.to_excel(xw, sheet_name="Monthly revenue", index=False)
        if d["Product"].nunique() > 1:
            prod = d.groupby("Product").agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum")) \
                .sort_values("Revenue", ascending=False).round(0).reset_index()
            prod.to_excel(xw, sheet_name="Products", index=False)
        if seg is not None:
            s = seg[["Customer", "Segment", "Recency", "Frequency", "Monetary"]].copy()
            s["Segment"] = s["Segment"].astype(str)
            s["Monetary"] = s["Monetary"].round(0)
            s.columns = ["Customer", "Segment", "Days since last order", "Orders", "Total spent"]
            s.to_excel(xw, sheet_name="Customer segments", index=False)
        w = scored.sort_values("ChurnRisk", ascending=False)[["Customer", "ChurnRisk", "Recency", "Frequency", "Monetary"]].copy()
        w["ChurnRisk"] = (w["ChurnRisk"] * 100).round(0)
        w["Monetary"] = w["Monetary"].round(0)
        w.columns = ["Customer", "Risk of leaving (%)", "Days since last order", "Orders", "Total spent"]
        w.to_excel(xw, sheet_name="Win-back list", index=False)
        for ws in xw.sheets.values():
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 80)
    return buf.getvalue()


# ------------------------------------------------------------------------- UI
st.markdown(CSS, unsafe_allow_html=True)
ss = st.session_state
ss.setdefault("mode", None)
ss.setdefault("raw", None)

bar = st.columns([4, 1.9, 1.3, 1.5], vertical_alignment="center")
bar[0].markdown(f'<div class="brand">{LOGO_SVG}Shop Insights</div>', unsafe_allow_html=True)
with bar[2].popover("⚙️ Settings"):
    st.text_input("Currency symbol", "₹", key="money")
    st.text_input("Shop name", "", key="shop", placeholder="Used in the win-back message")
money = ss.get("money", "₹") or "₹"
if ss.mode and bar[3].button("Change data", width="stretch"):
    ss.mode, ss.raw = None, None
    st.rerun()

# ---- landing page -----------------------------------------------------------
if ss.mode is None:
    hero("Know your shop. Know your customers.",
         "Turn raw sales data into customer insights, trends and a clear weekly action plan.", tag="Smart insights for every shop")
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("#### Try a demo shop")
        st.markdown("<span class='muted'>Explore every feature with a made-up grocery shop: 1,200 customers over two years.</span>",
                    unsafe_allow_html=True)
        st.write("")
        if st.button("Open the demo", type="primary", width="stretch"):
            ss.mode = "demo"
            st.rerun()
    with right.container(border=True):
        st.markdown("#### Use your own sales file")
        up = st.file_uploader("CSV or Excel, one row per item sold", type=["csv", "xlsx", "xls"])
        template = "OrderID,OrderDate,CustomerID,Product,Quantity,UnitPrice\nO1,2026-01-05,C1,Tea,2,120\n"
        st.download_button("Download a sample file", template, "template.csv", "text/csv")
        if up is not None:
            try:
                ss.raw, ss.mode = read_upload(up), "upload"
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"We could not read that file: {e}")
    st.write("")
    tiles = [("💡", "Plain-English insights", "Know your best day, best product and your most valuable customers."),
             ("🧩", "Customer groups", "Champions, Loyal, At Risk and Lost, each with a suggested action."),
             ("🔔", "Win-back list", "A prediction of who is about to stop buying, plus a message to send."),
             ("📥", "Excel report", "Download everything to share with your team or your accountant.")]
    for col, (ic, t, tx) in zip(st.columns(4), tiles):
        col.markdown(card(t, tx, icon=ic), unsafe_allow_html=True)
    st.write("")
    st.caption("🔒 Your file is processed in memory only and is never stored. "
               "Your file needs a date, a customer (ID, name or phone), an order or bill number, and a price.")
    st.stop()

# ---- dashboard --------------------------------------------------------------
raw = demo_data() if ss.mode == "demo" else ss.raw
if raw is None:
    ss.mode = None
    st.rerun()

hero_slot = st.container()
cols_slot = st.container()
cols = list(raw.columns)
guesses = {k: guess(cols, k) for k in ALIASES}
problem = len({guesses[k] for k in REQUIRED}) < 4 or any(guesses[k] is None for k in REQUIRED)

if ss.mode == "upload":
    with cols_slot:
        with st.expander("Check how we read your file", expanded=problem):
            st.caption("We matched your columns automatically. Change any that look wrong.")
            none = "(none)"

            def pick(label, key, required):
                g = guesses[key]
                options = cols if required else [none] + cols
                v = st.selectbox(label, options, index=options.index(g) if g in options else 0)
                return None if v == none else v

            a, b = st.columns(2)
            with a:
                mapping = {"date": pick("Date", "date", True),
                           "customer": pick("Customer", "customer", True),
                           "order": pick("Order or bill number", "order", True),
                           "price": pick("Price (per unit, or total if no quantity)", "price", True)}
            with b:
                mapping.update({"quantity": pick("Quantity (optional)", "quantity", False),
                                "product": pick("Product (optional)", "product", False),
                                "place": pick("City or country (optional)", "place", False)})
else:
    mapping = guesses

if len({mapping[k] for k in REQUIRED}) < 4:
    with cols_slot:
        st.warning("Each required field (date, customer, order, price) needs its own column. "
                   "Please pick them in the box above.")
    st.stop()

df, dropped = prepare(raw, mapping)
if len(df) < 20:
    with cols_slot:
        st.error("Fewer than 20 valid rows after cleaning. Check the column choices above and your date format.")
    st.stop()
if dropped:
    cols_slot.caption(f"{dropped:,} rows were ignored (missing values, returns or zero prices).")

seg = rfm_segments(df)
_, _, scored90 = churn_scores(df, 90)
ins = insights(df, seg, money)
dmin, dmax = df["Date"].min(), df["Date"].max()

# hero
n_cust, total_rev = df["Customer"].nunique(), df["Revenue"].sum()
title = f"{n_cust:,} customers spent {money}{fmt(total_rev, money)}."
if seg is not None:
    n_risk = int(seg["Segment"].isin(["At Risk", "Lost"]).sum())
    title += f" {n_risk:,} of them are at risk of leaving or already gone."
if ss.mode == "demo":
    tag, sub = "Demo data", "This is a made-up shop. Go back with “Change data” to upload your own file and see your real numbers."
else:
    tag, sub = "", f"Based on {len(df):,} sales rows from {dmin:%d %b %Y} to {dmax:%d %b %Y}."
with hero_slot:
    hero(title, sub, tag=tag, art=sparkline(df))

# report button in the top bar
bar[1].download_button("⬇️ Download report", build_report(df, seg, scored90, money, tuple(t for _, t in ins)),
                       "shop-insights-report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary", width="stretch")

# KPIs
orders = df["Order"].nunique()
last30, _, ch = trend(df)
note = f"{money}{fmt(last30, money)} in the last 30 days"
if ch is not None:
    note += f' <span class="{"up" if ch >= 0 else "down"}">{"▲" if ch >= 0 else "▼"} {abs(ch):.0%}</span>'
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi("Total revenue", f"{money}{fmt(total_rev, money)}", note, strong=True), unsafe_allow_html=True)
k2.markdown(kpi("Orders", f"{orders:,}", "separate bills or invoices"), unsafe_allow_html=True)
k3.markdown(kpi("Customers", f"{n_cust:,}", "who bought at least once"), unsafe_allow_html=True)
k4.markdown(kpi("Average order", f"{money}{fmt(total_rev / max(orders, 1), money)}", "spent per order"), unsafe_allow_html=True)

# weekly to-do
st.markdown("#### Do this week")
at_risk = scored90[scored90["ChurnRisk"] >= 0.5]
day_rev = df.groupby(df["Date"].dt.day_name())["Revenue"].sum()
todo = [card(f"Win back {len(at_risk):,} customers",
             f"They have spent {money}{fmt(at_risk['Monetary'].sum(), money)} with you and may have stopped buying. "
             f"The Win-back tab has the list and a message to send.", AMBER)]
todo.append(card(f"Fill your quiet {day_rev.idxmin()}",
                 f"{day_rev.idxmin()} brings the least sales. A small offer that day can even out your week.", LEAF))
if df["Product"].nunique() > 1:
    prod = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
    todo.append(card(f"Keep {prod.index[0]} in stock",
                     f"It brings {prod.iloc[0] / total_rev:.0%} of your sales, more than any other product.", TEAL))
elif seg is not None:
    n_champ = int((seg["Segment"] == "Champions").sum())
    todo.append(card(f"Thank your {n_champ:,} best customers",
                     "A personal thank-you or small gift keeps your Champions coming back.", TEAL))
for col, html in zip(st.columns(len(todo)), todo):
    col.markdown(html, unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Sales", "Customers", "Win-back"])

with tab1:
    a, b = st.columns([3, 2])
    with a.container(border=True):
        st.markdown("#### What your data says")
        rows = "".join(f'<div class="insight"><div class="ico">{ic}</div><div>{bold(t)}</div></div>' for ic, t in ins)
        st.markdown(rows, unsafe_allow_html=True)
    if seg is not None:
        summary = seg.groupby("Segment", observed=True).agg(Customers=("Customer", "count")).reset_index()
        pie = px.pie(summary, names="Segment", values="Customers", hole=0.55, title="Customer groups",
                     color="Segment", color_discrete_map=SEG_COLORS)
        pie.update_traces(textinfo="percent", sort=False)
        with b.container(border=True):
            st.plotly_chart(style(pie, height=420), width="stretch")

with tab2:
    monthly = df.set_index("Date").resample("MS")["Revenue"].sum().reset_index()
    fig = px.area(monthly, x="Date", y="Revenue", title="Monthly revenue")
    fig.update_traces(line=dict(color=TEAL, width=3), fillcolor="rgba(15,76,74,0.10)")
    style(fig).update_yaxes(tickprefix=money)
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    wd = df.groupby(df["Date"].dt.day_name())["Revenue"].sum().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index()
    wd.columns = ["Day", "Revenue"]
    with a.container(border=True):
        st.plotly_chart(style(px.bar(wd, x="Day", y="Revenue", title="Revenue by weekday"),
                              bar_color=TEAL).update_yaxes(tickprefix=money), width="stretch")
    if df["Product"].nunique() > 1:
        top = df.groupby("Product")["Revenue"].sum().nlargest(10).reset_index()
        f2 = style(px.bar(top, x="Revenue", y="Product", orientation="h", title="Top 10 products"), bar_color=LEAF)
        f2.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)").update_xaxes(tickprefix=money)
        with b.container(border=True):
            st.plotly_chart(f2, width="stretch")
    if df["Date"].dt.hour.nunique() > 1:
        df2 = df.assign(Hour=df["Date"].dt.hour, Day=df["Date"].dt.day_name())
        heat = df2.groupby(["Day", "Hour"])["Order"].nunique().reset_index()
        f3 = px.density_heatmap(heat, x="Hour", y="Day", z="Order", title="When customers order",
                                category_orders={"Day": ["Sunday", "Saturday", "Friday", "Thursday",
                                                         "Wednesday", "Tuesday", "Monday"]})
        f3.update_coloraxes(colorscale=[[0, "#EAF4F1"], [1, TEAL]], colorbar_title=None)
        with st.container(border=True):
            st.plotly_chart(style(f3), width="stretch")
    if df["Place"].nunique() > 1:
        pl = df.groupby("Place")["Revenue"].sum().nlargest(10).reset_index()
        with st.container(border=True):
            st.plotly_chart(style(px.bar(pl, x="Place", y="Revenue", title="Revenue by place"),
                                  bar_color=AMBER).update_yaxes(tickprefix=money), width="stretch")

with tab3:
    if seg is None:
        st.info("You need at least 8 customers for customer groups.")
    else:
        summary = seg.groupby("Segment", observed=True).agg(
            Customers=("Customer", "count"), Revenue=("Monetary", "sum")).reset_index()
        info = summary.set_index("Segment").to_dict("index")
        actions = {
            "Champions": "Buy often, recently, spend a lot. Reward them and ask for referrals.",
            "Loyal": "Regular, good spenders. Offer bundles and loyalty points.",
            "At Risk": "Used to buy, now slipping. Send a reminder or a small offer.",
            "Lost": "Haven't bought for a long time. Try one win-back message, then let go.",
        }
        for col, name in zip(st.columns(4), ["Champions", "Loyal", "At Risk", "Lost"]):
            s = info.get(name, {"Customers": 0, "Revenue": 0})
            col.markdown(
                f'<div class="seg" style="border-top:5px solid {SEG_COLORS[name]}">'
                f'<div class="seg-name">{name}</div>'
                f'<div class="seg-stat">{int(s["Customers"]):,} customers</div>'
                f'<div class="muted">{money}{fmt(s["Revenue"], money)} spent</div>'
                f'<div class="seg-do">{actions[name]}</div></div>', unsafe_allow_html=True)
        st.write("")
        bar_fig = px.bar(summary, x="Segment", y="Revenue", title="Revenue per group",
                         color="Segment", color_discrete_map=SEG_COLORS)
        with st.container(border=True):
            st.plotly_chart(style(bar_fig).update_layout(showlegend=False).update_yaxes(tickprefix=money),
                            width="stretch")
        st.download_button("Download customer groups (CSV)", seg.to_csv(index=False),
                           "customer_groups.csv", "text/csv")

with tab4:
    window = st.slider("Treat a customer as gone after this many days without buying", 30, 180, 90, 15)
    metrics, importance, scored = churn_scores(df, window)
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(kpi("Recall", f"{metrics['Recall']:.0%}", "leavers the model catches", strong=True), unsafe_allow_html=True)
        m2.markdown(kpi("Precision", f"{metrics['Precision']:.0%}", "flagged who really left"), unsafe_allow_html=True)
        m3.markdown(kpi("ROC-AUC", f"{metrics['ROC-AUC']:.2f}", "1.00 is perfect"), unsafe_allow_html=True)
        m4.markdown(kpi("Customers who left", f"{metrics['Churn rate']:.0%}", f"no order in {window} days"), unsafe_allow_html=True)
        st.caption(f"A Random Forest model trained on {metrics['Customers']:,} customers and tested on a held-out 25% "
                   f"predicts who will not buy in the next {window} days.")
        imp = importance.rename(index={"Recency": "Days since last order", "Frequency": "Number of orders",
                                       "Monetary": "Total spent", "Tenure": "Days as customer",
                                       "Items": "Items bought", "AvgOrderValue": "Average order"})
        with st.container(border=True):
            st.plotly_chart(style(px.bar(imp, orientation="h", title="What predicts a customer leaving"),
                                  bar_color=TEAL).update_layout(showlegend=False), width="stretch")
    else:
        st.info("Not enough history for a prediction model (needs 100+ customers and some who stopped buying), "
                "so this list uses a simple rule: the longer since the last order, the higher the risk.")

    st.markdown("#### Customers to contact first")
    st.caption("Biggest spenders who are most likely to be gone.")
    risky = scored[scored["ChurnRisk"] >= 0.5].sort_values("Monetary", ascending=False).head(50).copy()
    risky["RiskPct"] = risky["ChurnRisk"] * 100
    st.dataframe(
        risky[["Customer", "RiskPct", "Recency", "Frequency", "Monetary"]],
        width="stretch", hide_index=True,
        column_config={
            "Customer": "Customer",
            "RiskPct": st.column_config.ProgressColumn("Risk of leaving", min_value=0, max_value=100, format="%d%%"),
            "Recency": st.column_config.NumberColumn("Days since last order"),
            "Frequency": st.column_config.NumberColumn("Orders"),
            "Monetary": st.column_config.NumberColumn("Total spent", format="%.0f"),
        })
    st.download_button("Download full win-back list (CSV)",
                       scored.sort_values("ChurnRisk", ascending=False).to_csv(index=False),
                       "winback_list.csv", "text/csv")

    st.markdown("#### Send a win-back message")
    shop = (ss.get("shop") or "").strip() or "our shop"
    default_msg = (f"Hi! It's been a while since your last visit to {shop}. We miss you! "
                   f"Show this message on your next visit and get 10% off. See you soon!")
    st.text_area("Edit the message, then send it on WhatsApp or SMS to the customers above",
                 default_msg, height=110, key=f"msg_{shop}")

st.write("")
st.caption("Built with Python, scikit-learn, Plotly and Streamlit. Your data stays in memory and is never stored.")