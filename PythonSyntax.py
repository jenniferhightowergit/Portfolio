# =========================
# CodeSignal DAA - Python/Pandas Cheat Sheet (CSV Analysis)
# =========================

import pandas as pd
import numpy as np






# ---------- Load / Inspect ----------
df = pd.read_csv("file.csv")                 # basic
# df = pd.read_csv("file.csv", encoding="utf-8")   # if needed
df.head()
df.tail()				# last 5 rows
df.shape                                    # (rows, cols)
df.columns
df.dtypes
df.info()
df.describe(include="all")                  # numeric + object summary
df.isna().sum()

# ---------- Missing / Duplicates ----------
df.isna().sum().sort_values(ascending=False)
df_join["transaction_ts"].sort_values().head(100)
df.duplicated() #returns a boolean Series: True for rows that are duplicates of a previous row
df.duplicated().sum() #.sum() counts True values (because True behaves like 1).
df[df.duplicated()]                         # view dup rows
df.drop_duplicates()




df[["amount", "revenue"]].isna().sum(). #count nulls
df = df.dropna(subset=["amount", "revenue"])  #removue row nulls
(df[["amount", "revenue"]].isna().mean() * 100).round(2) # get precentile of nulls

#note you would only fill nulls with mean or median if you are modeling data, not for event data
df["amount_filled"] = df["amount"].fillna(df["amount"].mean()) #fill null with mean
df["amount_filled"] = df["amount"].fillna(df["amount"].median()) #fill null with median <-- prefered for modeling data

#better approach, don't remove columns, but make new ones
df["amount_imputed"] = df["amount"].fillna(df["amount"].median())
df["amount_was_null"] = df["amount"].isna().astype(int)




#convert strings to numbers
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")


#checks (duplicates / uniqueness)
df.shape
df[["txn_id"]].duplicated().sum()
df.groupby("customer_id").size().describe()
df["customer_id"].nunique()


#Clean currency/commas fast

df["amount"] = (
    df["amount"].astype(str)
      .str.replace(r"[$,]", "", regex=True)
      .replace("nan", np.nan)
)
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")


#----------------quick lookups

# top n per group

# top 2 products per category by revenue, deterministic tie-break by product_id
tmp = (df.groupby(["category","product_id"], as_index=False)
         .agg(revenue=("revenue","sum")))
tmp = tmp.sort_values(["category","revenue","product_id"], ascending=[True, False, True])
tmp["rn"] = tmp.groupby("category").cumcount() + 1
top2 = tmp[tmp["rn"] <= 2]
top2


#Confidence interval quickie
# 95% CI for conversion rate p (event-level)
p = df["converted"].mean()
n = df["converted"].notna().sum()
se = (p*(1-p)/n) ** 0.5
ci95 = (p - 1.96*se, p + 1.96*se)
p, ci95







# ---------- Basic Column Ops ----------
df["col"]                                   # series
df[["a","b"]]                               # subset
df.rename(columns={"old":"new"}, inplace=True)
df["col"].astype("string")                  # or "Int64" nullable int
df["col"] = pd.to_numeric(df["col"], errors="coerce") #coerce, If a value can’t be converted to a number, pandas will replace it with NaN instead of throwing an error.

# ---------- Dates / Times ----------
df["ts"] = pd.to_datetime(df["ts"], errors="coerce")   # parse timestamps
df["date"] = df["ts"].dt.date
df["month"] = df["ts"].dt.to_period("M").astype(str)
df = df.sort_values(["id","ts"])

#for string or object dates
trans_df["transaction_ts"] = pd.to_datetime(
    trans_df["transaction_ts"],
    errors="coerce"
)

df["date"] = df["ts"].dt.to_period("D").dt.to_timestamp()
df["month"] = df["ts"].dt.to_period("M").dt.to_timestamp()
df["year"] = df["ts"].dt.to_period("Y").dt.to_timestamp()


# ---------- Filtering ----------
df[df["amount"] > 0]
df[(df["a"] == "x") & (df["b"].isna())]
df[df["col"].isin(["A","B","C"])]  # same as SQL WHERE col IN ('A','B','C')

# ---------- Value counts / cardinality ----------
df["col"].value_counts(dropna=False)
df["col"].nunique(dropna=True)

# ---------- Aggregations ----------
df["amount"].sum()
df["amount"].mean()
df["amount"].median()
df["amount"].min(), df["amount"].max()
df["amount"].std(ddof=1)                     # sample std (like most stats)
df["amount"].quantile(0.9)                   # 90th percentile

# -----------Calulations ---------------
# MOM
m = df.groupby("month", as_index=False).agg(revenue=("amount","sum"))

m = m.sort_values("month")
m["revenue_prev_month"] = m["revenue"].shift(1)
m["mom_abs"] = m["revenue"] - m["revenue_prev_month"]
m["mom_pct"] = (m["revenue"] / m["revenue_prev_month"]) - 1

"""
SQL equivilant:

WITH monthly AS (
  SELECT
    DATE_TRUNC('month', ts) AS month,
    SUM(amount) AS revenue
  FROM df
  GROUP BY 1
),
calc AS (
  SELECT
    month,
    revenue,
    LAG(revenue, 1)  OVER (ORDER BY month)  AS revenue_prev_month,
    LAG(revenue, 12) OVER (ORDER BY month)  AS revenue_prev_year,
    SUM(revenue) OVER (ORDER BY month
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS revenue_running
  FROM monthly
)
SELECT
  month,
  revenue,
  revenue_prev_month,
  revenue - revenue_prev_month AS mom_abs,
  (revenue / NULLIF(revenue_prev_month, 0)) - 1 AS mom_pct,
  revenue_prev_year,
  revenue - revenue_prev_year AS yoy_abs,
  (revenue / NULLIF(revenue_prev_year, 0)) - 1 AS yoy_pct,
  revenue_running
FROM calc
ORDER BY month;

"""

# YOY
m["revenue_prev_year"] = m["revenue"].shift(12)     # if monthly series
m["yoy_abs"] = m["revenue"] - m["revenue_prev_year"]
m["yoy_pct"] = (m["revenue"] / m["revenue_prev_year"]) - 1


# Running total cumulative
m["revenue_running"] = m["revenue"].cumsum()

# By segment
ms = df.groupby(["segment","month"], as_index=False).agg(revenue=("amount","sum"))
ms = ms.sort_values(["segment","month"])
ms["running_revenue"] = ms.groupby("segment")["revenue"].cumsum()

# Rolling windows baseline vs anomaly detection
df = df.sort_values(["customer_id","ts"])
df["roll7_avg"] = (
    df.groupby("customer_id")["amount"]
      .rolling(window=7, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

"""
SQL evuivilant running segment
WITH seg_month AS (
  SELECT
    segment,
    DATE_TRUNC('month', ts) AS month,
    SUM(amount) AS revenue
  FROM df
  GROUP BY 1, 2
)
SELECT
  segment,
  month,
  revenue,
  SUM(revenue) OVER (
    PARTITION BY segment
    ORDER BY month
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS running_revenue
FROM seg_month
ORDER BY segment, month;



Rolling 7 average” by customer (window=7 rows)

SELECT
  customer_id,
  ts,
  amount,
  AVG(amount) OVER (
    PARTITION BY customer_id
    ORDER BY ts
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS roll7_avg
FROM df
ORDER BY customer_id, ts;


"""


"""
Conversion rate
If you have funnel events:

visits table or column

converted as 0/1
"""


#----------------------Conversion rate-----------------

conv = df.groupby("channel", as_index=False).agg(
    visits=("visitor_id","nunique"),
    conversions=("converted","sum")
)
conv["conversion_rate"] = conv["conversions"] / conv["visits"]

df["converted"].mean()  # if converted is 0/1 and each row is a visitor

conv


#event - everything - “visits / events” → event-level
#customer by customer - “customers / users” → user-level
#account by account - “customers / users” → user-level

#event level conversion
#event level conversion

event_conv = (
    df.groupby("ab_group")["converted"].mean()
)
event_conv

#not event level, but for unique customer, customers, accounts, or ever converted
#“Out of all events (rows), what fraction converted?”
df.groupby(["ab_group", "customer_id"])["converted"].max().groupby("ab_group").mean()



# per-customer conversion rate (basically “what percent of their events converted”),
conv = (
    trans_df
    .groupby("customer_id", as_index=False)
    .agg(
        visits=("customer_id", "count"),
        conversions=("converted", "sum")
    )
)

conv["conversion_rate"] = conv["conversions"] / conv["visits"]

print(conv.head())



#Customer-level conversion rate by group
cust = (
    df.groupby(["ab_group", "customer_id"])["converted"]
      .max()
      .reset_index()
)

cust_conv = (
    cust.groupby("ab_group")["converted"]
        .mean()   # mean of 0/1 customers = customer-level conversion rate
        .reset_index(name="cust_conversion_rate")
)
cust_conv




# conversion with groupby
conv_events = (
    df_join
    .groupby(["credit_score_bin", "income_bin"], observed=True)   # <- also silences warning
    .agg(
        events=("converted", "count"),
        conversions=("converted", "sum")
    )
    .reset_index()
)

conv_events["conversion_rate"] = conv_events["conversions"] / conv_events["events"]

conv_events.head()



#------------------- ARPU ---------------------




#revenue per customer
# ARPU = Average Revenue Per User (Customer)
#total revenue / number of unique customers

arpu_check = (
    df.groupby(["ab_group", "customer_id"])["revenue"]
      .sum()
      .groupby("ab_group")
      .mean()
      .reset_index(name="ARPU")
)

arpu_check




"""
Risk scoring pattern (loan eligibility style)

Example features:
income
dti (debt-to-income)
delinq_12m
utilization
credit_score
"""

df["risk_score"] = 0

df.loc[df["credit_score"] < 620, "risk_score"] += 40
df.loc[df["dti"] > 0.4, "risk_score"] += 25
df.loc[df["utilization"] > 0.8, "risk_score"] += 20
df.loc[df["delinq_12m"] >= 1, "risk_score"] += 30
df.loc[df["income"] < 35000, "risk_score"] += 10

df["risk_band"] = pd.cut(
    df["risk_score"],
    bins=[-1, 29, 59, 1000],
    labels=["Low", "Medium", "High"]
)

"""
SQL evuivilent: Risk score, risk band, approve flag (rule-based scoring)

WITH scored AS (
  SELECT
    d.*,
    (
      CASE WHEN credit_score < 620 THEN 40 ELSE 0 END
    + CASE WHEN dti > 0.4 THEN 25 ELSE 0 END
    + CASE WHEN utilization > 0.8 THEN 20 ELSE 0 END
    + CASE WHEN delinq_12m >= 1 THEN 30 ELSE 0 END
    + CASE WHEN income < 35000 THEN 10 ELSE 0 END
    ) AS risk_score
  FROM df d
)
SELECT
  scored.*,
  CASE
    WHEN risk_score <= 29 THEN 'Low'
    WHEN risk_score <= 59 THEN 'Medium'
    ELSE 'High'
  END AS risk_band,
  CASE WHEN risk_score < 60 THEN 1 ELSE 0 END AS approve
FROM scored;



"""


#rolling average

#You want a 7-day rolling average of transaction amounts, ordered by time, per customer.
#Make sure data is sorted (CRITICAL)

"""
A rolling average (also called a moving average) is:

An average computed over a sliding window of recent observations, recalculated at each point in time to smooth short-term fluctuations and highlight underlying trends.

Plain-English version

Instead of averaging everything at once, you:

pick a window (e.g., last 7 days, last 10 transactions)

move that window forward one step at a time

recompute the average each time

So the average “rolls” forward as new data arrives.

"""

df = df.sort_values("transaction_ts")

df["rolling_7_avg"] = df["amount"].rolling(window=7, min_periods=1).mean()



#with grouping
df = df.sort_values(["customer_id", "transaction_ts"])

df["rolling_7_avg"] = (
    df
      .groupby("customer_id")["amount"]
      .rolling(window=7, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)




df["approve"] = np.where(df["risk_score"] < 60, 1, 0)

#Delinquency rate:
delinq_rate = df["is_delinquent"].mean()   # if 0/1

#Charge-off rate:
chargeoff_rate = df["is_chargeoff"].mean()

#Loss rate / net loss:
df["net_loss"] = df["chargeoff_amt"] - df["recovery_amt"]

# Average balance:
avg_bal = df.groupby("customer_id")["balance"].mean()




# ---------- Groupby calculation----------

overall_stats = df["amount"].agg(
    count="count",
    total="sum",
    mean="mean",
    median="median",
    std="std"
)

overall_stats


g = df.groupby("segment")["amount"].agg(["count","sum","mean","median", "std", "sum", "min", "max"]).reset_index()
g



# ----------   calc for all object datatypes

# by object datatype / column!!!

import numpy as np
import pandas as pd

# object + numeric columns
obj_cols = df.select_dtypes(include="object").columns.tolist()
num_col = "amount"   # change if your measure column is named differently

rows = []

for col in obj_cols:
    g = df.groupby(col)[num_col].agg(
        count="count",
        sum="sum",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max"
    )

    rows.append({
        "Group": col,
        "count": g["count"].sum(),
        "sum": g["sum"].sum(),
        "mean": g["mean"].mean(),
        "median": g["median"].median(),
        "std": g["std"].mean(),
        "min": g["min"].min(),
        "max": g["max"].max()
    })

summary_table = pd.DataFrame(rows)
summary_table



#multiple groupings
bin_stats = (
    df_join
    .groupby(["credit_score_bin", "income_bin"], observed=True)
    .agg(
        txn_count=("amount", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        median_amount=("amount", "median"),
        std_amount=("amount", "std")
    )
    .reset_index()
)

bin_stats


"""
Similar to SQL:
SELECT segment, channel,
       COUNT(id) AS n,
       SUM(amount) AS revenue,
       AVG(amount) AS avg_amount
FROM df
GROUP BY segment, channel;

"""


df.groupby(["ab_group", "customer_id"])["converted"].max().groupby("ab_group").mean()


# ---------- Pivot (like Excel pivot tables) ----------
pv = df.pivot_table(
index="segment", # SQL: group by
columns="channel", # SQL Case when - this is the pivot
values="amount", #SQL case Then result
aggfunc="sum",  #calculation function
fill_value=0)
pv

"""
SQL evuivilant:
SELECT
  segment,
  SUM(CASE WHEN channel = 'email'  THEN amount ELSE 0 END) AS email_amount,
  SUM(CASE WHEN channel = 'search' THEN amount ELSE 0 END) AS search_amount,
  SUM(CASE WHEN channel = 'social' THEN amount ELSE 0 END) AS social_amount
FROM df
GROUP BY segment;


"""

# ---------- Joins / Merges ----------
# Inner / left joins like SQL
df_join = df_left.merge(df_right, on="key", how="left")
df_join = df_left.merge(df_right, on=["k1","k2"], how="inner")

# Join with different key names
df_join = df_left.merge(df_right, left_on="cust_id", right_on="customer_id", how="left")

# ---------- Window-ish patterns ----------
# Row number within group
df["rn"] = df.sort_values("ts").groupby("cust_id").cumcount() + 1

# Lag within group
df["prev_amount"] = df.sort_values("ts").groupby("cust_id")["amount"].shift(1)

# Rolling window (e.g., 7-day moving average by customer)
# Requires time index or use rolling on sorted rows
df = df.sort_values(["cust_id","ts"])
df["roll7_mean"] = df.groupby("cust_id")["amount"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)

# ---------- Simple anomaly checks ----------




# Outliers via z-score (rough)
df["z"] = (df["amount"] - df["amount"].mean()) / df["amount"].std(ddof=1)
df[np.abs(df["z"]) > 3]



# ---------- String cleaning ----------
df["col"] = df["col"].astype("string").str.strip().str.lower()
df["col"].str.contains("foo", na=False)
df["col"].str.extract(r"(\d+)")             # regex capture. This pulls out the first group of digits from each string using regex. 

# ---------- Safe division ----------
def safe_div(n, d):
    return np.where(d == 0, np.nan, n / d)

# ---------- Quick export (if allowed/needed) ----------
# df.to_csv("out.csv", index=False)


#-----------------IQR outliers-------------------------

#note: Make sure you do pd.to_numeric(..., errors="coerce") on amount before IQR/z-score checks.
#Tukey IQR rule:

#below returns a list of outliers
amount = pd.to_numeric(df["amount"], errors="coerce")

q1, q3 = amount.quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
out_iqr = df[(amount < lo) | (amount > hi)]
out_iqr





import matplotlib.pyplot as plt

#for one group
df_join.boxplot(column="amount", by="segment", grid=False)
plt.title("Transaction Amount by Segment")
plt.suptitle("")   # removes auto title
plt.xlabel("Segment")
plt.ylabel("Amount")
plt.show()


#binning
df_join["credit_score_bin"] = pd.cut(
    df_join["credit_score"],
    bins=[300, 580, 670, 740, 800, 850],
    labels=["Poor", "Fair", "Good", "Very Good", "Excellent"]
)

df_join.boxplot(column="amount", by="credit_score_bin", grid=False)
plt.title("Transaction Amount by Credit Score Band")
plt.suptitle("")
plt.show()




#month year buckets
df_join["month"] = df_join["transaction_ts"].dt.to_period("M").astype(str)
df_join.boxplot(column="amount", by="month", grid=False, rot=45)
plt.title("Transaction Amount by Month")
plt.suptitle("")
plt.show()


#-------------------IQR boxplot all objects -------------------

## calulate IQR

import pandas as pd
import numpy as np

def get_numeric_cols(df, exclude=None):
    """Return a list of numeric columns (int/float), minus any excluded ones."""
    exclude = set(exclude or [])
    cols = df.select_dtypes(include=[np.number]).columns
    return [c for c in cols if c not in exclude]


import matplotlib.pyplot as plt
import math

def boxplot_grid(df, cols=None, exclude=None, showfliers=True, ncols=3):
    if cols is None:
        cols = get_numeric_cols(df, exclude=exclude)

    n = len(cols)
    if n == 0:
        print("No numeric columns found.")
        return

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1)

    for ax, c in zip(axes, cols):
        df.boxplot(column=c, ax=ax, showfliers=showfliers)
        ax.set_title(c)

    # hide unused axes
    for ax in axes[len(cols):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

boxplot_grid(df)


#----------------- remove outliers --------------------




# remove outliers

def iqr_bounds(s, k=1.5):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return lo, hi

def remove_outliers_iqr(df, cols=None, exclude=None, k=1.5, how="any"):
    """
    Remove rows containing IQR outliers.
    how="any": drop row if it's an outlier in ANY selected column (stricter)
    how="all": drop row only if it's an outlier in ALL selected columns (looser)
    """
    if cols is None:
        cols = get_numeric_cols(df, exclude=exclude)

    if len(cols) == 0:
        return df.copy()

    mask_inlier = pd.Series(True, index=df.index)

    # build per-column inlier masks
    col_masks = []
    for c in cols:
        s = df[c]
        lo, hi = iqr_bounds(s.dropna(), k=k)
        col_inlier = s.between(lo, hi) | s.isna()  # keep NaNs
        col_masks.append(col_inlier)

    col_masks = pd.concat(col_masks, axis=1)

    if how == "any":
        # keep rows that are inliers for ALL cols (drop if outlier in any col)
        mask_inlier = col_masks.all(axis=1)
    elif how == "all":
        # drop rows only if outlier in all cols => keep if inlier in at least one col
        mask_inlier = col_masks.any(axis=1)
    else:
        raise ValueError("how must be 'any' or 'all'")

    return df.loc[mask_inlier].copy()


df = remove_outliers_iqr(df)
df.shape, df.shape



# using z-score to remove outliers

from scipy.stats import zscore

# compute z-scores for amount
df["z_amount"] = zscore(df["amount"], nan_policy="omit")
df_no_outliers = df[df["z_amount"].abs() <= 3]




#returns a list of outliers outside a standard dev of 3
from scipy.stats import zscore
z = zscore(amount.dropna())
out_z = df.loc[amount.dropna().index[np.abs(z) > 3]]
out_z





#-------------t-test

# When to use a t-test vs proportions test
# Metric	Test
# Conversion rate (0/1)	t-test (acceptable) or proportion test
# Revenue / ARPU	Two-sample t-test
# Large samples	t-test is fine
# Skewed revenue	Mention assumption / robustness


"""
Example answer (perfect length)

“I tested the null hypothesis that there is no difference in conversion rate between A and B using a two-sample t-test. The alternative hypothesis is that the conversion rates differ. Based on the p-value, I [reject / fail to reject] the null hypothesis at the 5% significance level.”
"""

#Customer-level t-test:
cust_rev = (
    df.groupby(["ab_group","customer_id"])["revenue"]
      .sum()
      .reset_index()
)

A = cust_rev.loc[cust_rev["ab_group"]=="A", "revenue"]
B = cust_rev.loc[cust_rev["ab_group"]=="B", "revenue"]

t_stat, p_val = ttest_ind(A, B, equal_var=False)
t_stat, p_val



# SEgemtation: “Prime only” slice (as requested):
prime = df[df["segment"]=="Prime"]

cust_rev_prime = (
    prime.groupby(["ab_group","customer_id"])["revenue"]
         .sum()
         .reset_index()
)
A = cust_rev_prime.loc[cust_rev_prime["ab_group"]=="A", "revenue"]
B = cust_rev_prime.loc[cust_rev_prime["ab_group"]=="B", "revenue"]
ttest_ind(A, B, equal_var=False)



#comparing groups: Email vs Search statistical test
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# successes and trials per channel
tmp = df.groupby("channel")["converted"].agg(["sum","count"])
count = tmp["sum"].astype(int).values
nobs  = tmp["count"].astype(int).values

z_stat, p_val = proportions_ztest(count, nobs)
z_stat, p_val



#Two-sample t-test for ARPU (or revenue)

A = df.loc[df["ab_group"] == "A", "revenue"].dropna()
B = df.loc[df["ab_group"] == "B", "revenue"].dropna()

from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(A, B, equal_var=False)
t_stat, p_value

#interpret

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")



#B) t-test for conversion rate (0/1 column)

A = df.loc[df["ab_group"] == "A", "converted"]
B = df.loc[df["ab_group"] == "B", "converted"]

t_stat, p_value = ttest_ind(A, B, equal_var=False)
t_stat, p_value

# check mean 
A.mean(), B.mean()
