import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pricing Tool", layout="wide")

COLS = ["week", "store", "product", "list_price", "discount_pct", "competitor_price", "unit_cost", "orders"]
MAX_MOVE = 15  # price can move at most +/-15% from today's price


def load_data():
    f = st.sidebar.file_uploader("Use your own CSV (optional)", type="csv")
    if f is None:
        return pd.read_csv("sample_data.csv")
    df = pd.read_csv(f)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()
    return df


@st.cache_data
def fit(df):
    """Orders vs price paid, competitor price and season (log-log regression)."""
    out = {}
    for (s, p), g in df.groupby(["store", "product"]):
        paid = g.list_price * (1 - g.discount_pct / 100)
        X = np.column_stack([np.ones(len(g)), np.log(paid), np.log(g.competitor_price),
                             np.sin(2 * np.pi * (g.week - 1) / 52)])
        y = np.log(g.orders)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ beta
        cov = (res @ res / (len(g) - 4)) * np.linalg.inv(X.T @ X)
        out[(s, p)] = dict(a=beta[0], b=beta[1], c=beta[2], se=np.sqrt(cov[1, 1]),
                           comp=float(np.exp(np.log(g.competitor_price).mean())),
                           base=float(g.list_price.median()), cost=float(g.unit_cost.iloc[-1]),
                           true=float(g.true_elasticity.iloc[0]) if "true_elasticity" in g else None)
    return out


def result(m, price, disc=0, comp_chg=0, b=None):
    b = m["b"] if b is None else b
    q0 = np.exp(m["a"] + m["b"] * np.log(m["base"]) + m["c"] * np.log(m["comp"]))
    paid = price * (1 - disc / 100)
    q = q0 * (paid / m["base"]) ** b * (1 + comp_chg / 100) ** m["c"]
    return q, q * paid, q * (paid - m["cost"])  # orders, revenue, contribution


def best_price(m, comp_chg):
    prices = np.arange(round(m["base"] * (1 - MAX_MOVE / 100)), round(m["base"] * (1 + MAX_MOVE / 100)) + 1)
    return prices[np.argmax([result(m, p, 0, comp_chg)[2] for p in prices])]


rupee = lambda v: f"Rs {v:,.0f}"
change = lambda new, old: (new / old - 1) * 100

df = load_data()
models = fit(df)

store = st.sidebar.selectbox("Store", sorted(df.store.unique()))
product = st.sidebar.selectbox("Product", sorted(df["product"].unique()))
m = models[(store, product)]
base = m["base"]
price = st.sidebar.slider("New price (Rs)", int(base * .75), int(base * 1.25), int(base))
disc = st.sidebar.slider("Discount (%)", 0, 25, 0)
comp = st.sidebar.slider("Competitor price change (%)", -10, 10, 0)

st.title("Pricing Tool")
st.write(f"**{product}** at **{store}**. Current price: {rupee(base)}, cost per unit: {rupee(m['cost'])}.")

# what happens at the price you picked
now = result(m, base)
new = result(m, price, disc, comp)
with st.container(border=True):
    st.subheader("1. What happens at the price you chose")
    st.caption(f"New price {rupee(price)} with {disc}% discount, compared with today")
    c1, c2, c3 = st.columns(3)
    c1.metric("Orders per week", f"{new[0]:,.0f}", f"{change(new[0], now[0]):+.1f}%")
    c2.metric("Revenue per week", rupee(new[1]), f"{change(new[1], now[1]):+.1f}%")
    c3.metric("Contribution per week", rupee(new[2]), f"{change(new[2], now[2]):+.1f}%")

# recommendation
bp = best_price(m, comp)
rec = result(m, bp, 0, comp)
gain = rec[2] - now[2]
low, high = m["b"] - 1.96 * m["se"], m["b"] + 1.96 * m["se"]
lifts = [change(result(m, bp, 0, comp, b)[2], result(m, base, 0, 0, b)[2]) for b in (low, high)]

with st.container(border=True):
    st.subheader("2. Recommended price")
    left, right = st.columns([2, 3])
    with left:
        if bp == round(base):
            st.write(f"Keep the price at **{rupee(base)}**.")
        else:
            st.write(f"Change the price to **{rupee(bp)}** ({change(bp, base):+.1f}%).")
            st.write(f"Contribution goes up by about {rupee(gain)} per week ({change(rec[2], now[2]):+.1f}%).")
        st.write(f"The estimate could be off, so the realistic range for the change is "
                 f"{min(lifts):+.1f}% to {max(lifts):+.1f}%.")
    with right:
        prices = np.arange(int(base * .75), int(base * 1.25) + 1, 2)
        chart = pd.DataFrame({
            "Price": prices,
            "Revenue": [result(m, p, 0, comp)[1] / now[1] * 100 for p in prices],
            "Contribution": [result(m, p, 0, comp)[2] / now[2] * 100 for p in prices],
        }).set_index("Price")
        st.caption("Revenue and contribution at each price (current price = 100)")
        st.line_chart(chart, color=["#2563eb", "#f97316"], height=260)

# all stores
rows = []
for s in sorted(df.store.unique()):
    ms = models[(s, product)]
    p = best_price(ms, comp)
    row = {"Store": s, "Current price": rupee(ms["base"]), "Best price": rupee(p),
           "Contribution change": f"{change(result(ms, p, 0, comp)[2], result(ms, ms['base'])[2]):+.1f}%",
           "Price sensitivity": round(ms["b"], 2)}
    if ms["true"] is not None:
        row["Actual (synthetic data)"] = ms["true"]
    rows.append(row)
with st.container(border=True):
    st.subheader("3. Same product, all stores")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

with st.expander("Data and assumptions"):
    st.write("- Data is synthetic: 52 weeks of price, discount, competitor price and orders for each store and product.")
    st.write("- Price sensitivity is worked out from this history. A value of -1.5 means a 1% price rise loses about 1.5% of orders.")
    st.write("- Contribution = (price after discount - unit cost) x orders. The recommendation maximises this.")
    st.write(f"- Prices are limited to +/-{MAX_MOVE}% of the current price.")
    st.write("- Limitation: it assumes past customer behaviour continues. Test on a few stores before rolling out.")
    st.dataframe(df[(df.store == store) & (df["product"] == product)], hide_index=True)
    st.download_button("Download data", df.to_csv(index=False), "pricing_data.csv", "text/csv")
