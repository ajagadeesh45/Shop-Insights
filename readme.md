<img src="logo.svg" width="72" alt="Shop Insights logo">

## Shop Insights

Upload your shop's sales file (CSV or Excel) and understand your business in minutes: no coding, no paid software.

**Live app:** _add your Streamlit link here_

## Why I built it
Small shops keep sales records but can't afford analytics tools. This app turns a plain sales file into answers:
who are my best customers, which products sell, and who has stopped coming back.

## Features
- **Landing page** with a one-click demo and drag-and-drop upload (no sidebar, no filters to fiddle with)
- **"Do this week" action cards**: win back customers, fill the quietest day, keep the top product in stock
- **Plain-language insights**: e.g. "your top 20% of customers bring 48% of revenue"
- **Sales trends**: monthly revenue, best weekday, busiest hours, top products, sales by city
- **Customer segments (RFM)**: Champions, Loyal, At Risk, Lost, with suggested actions
- **Win-back list**: customers likely to stop buying, ranked by how much they spend (Random Forest churn model; falls back to a simple rule for small datasets)
- **Excel report download** (summary, monthly revenue, products, customer groups, win-back list)
- **Win-back message** you can edit and send on WhatsApp or SMS
- **Works with your file**: auto-detects columns, lets you correct them, and offers a template
- **Privacy**: the file is processed in memory only and never stored

## Tech
Python, pandas, scikit-learn (Random Forest), Plotly, Streamlit

## Your file needs (one row per item sold)
| Column | Required | Example names it recognises |
|---|---|---|
| Date | yes | OrderDate, InvoiceDate, Date |
| Customer | yes | CustomerID, Phone, Email |
| Order / bill number | yes | OrderID, InvoiceNo, Bill |
| Price | yes | UnitPrice, Price, Amount |
| Quantity, Product, City | optional | Qty, Item, City |

## Churn model
Churn = no purchase in the following N days (default 90). Features per customer: recency, frequency,
total spend, tenure, items bought, average order value. Trained on the past, tested on a held-out 25%.
Metrics (recall, precision, ROC-AUC) are shown live in the app for whatever data is loaded.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
The built-in demo data is synthetic (a made-up shop) so the app works without any upload.
Tested with the UCI Online Retail format too.
