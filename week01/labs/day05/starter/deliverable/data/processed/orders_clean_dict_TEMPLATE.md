# `orders_clean.parquet` — Data Dictionary  [STUDENT TEMPLATE]

**Cordwell Home & Hardware · Week 1 Capstone**

> Deliverable: fill in one row per clean-output column. For each, give the **type**,
> whether it is **nullable**, what it **means**, and any **cleaning caveat**
> (imputed? sentinel-filled? capped? canonicalized?). A data dictionary explains
> *meaning*, not just type — write it for "you in six weeks."

One row = one **order line**. The clean file has one row per `order_id`.

| Column | Type | Nullable | Description & cleaning caveats |
|---|---|---|---|
| `order_id` | string | No | TODO |
| `order_date` | datetime64[ns, UTC] | No | TODO |
| `store_id` | int64 | No | TODO |
| `store_region` | string | No | TODO (canonical set? how were nulls handled?) |
| `product_sku` | string | No | TODO |
| `product_category` | string | No | TODO |
| `product_name` | string | No | TODO (what normalization was applied?) |
| `quantity` | int64 | No | TODO (bounds? how were nulls/outliers handled?) |
| `quantity_capped` | bool | No | TODO |
| `unit_price` | float64 | No | TODO |
| `unit_price_capped` | bool | No | TODO |
| `discount_pct` | float64 | No | TODO |
| `channel` | string | No | TODO |

**Dropped from the raw file:** `updated_at` — TODO (why is it not in the clean output?)

**Provenance:** TODO (which script generated the raw data, which function cleaned it,
what validated it before writing?)
