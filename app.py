"""
E-Commerce Intelligence & Bulk Order Classifier  (English / Arabic)
-------------------------------------------------------------------
Run with:  streamlit run app.py

Structure
  1. Translations (TR) + language helpers
  2. Config & constants
  3. Styling helpers
  4. Data & model loading
  5. Statistics (separability, correlations)
  6. Chart builders
  7. Pages (Overview / Analytics / Classifier)

Language: chosen from the sidebar (key="lang"). Because Streamlit reruns the
script on every interaction, LANG is read from session_state at the very top
and every module-level constant below is built from it.
Internal data keys (feature names, "High/Moderate/Low", "Median" ...) always
stay in English so the cached computations are language independent; only the
*display* strings are translated.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, ks_2samp
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ════════════════════════════════════════════════════════════════════════════
# 1. TRANSLATIONS   key -> (English, Arabic)
# ════════════════════════════════════════════════════════════════════════════
TR: dict[str, tuple[str, str]] = {
    # --- general -----------------------------------------------------------
    "page_title": ("E-Commerce Intelligence & Bulk Classifier", "ذكاء التجارة الإلكترونية ومصنّف الطلبات بالجملة"),
    "language": ("🌐 Language", "🌐 اللغة"),
    "seg_bulk": ("Bulk / Wholesale", "جملة / تجار"),
    "seg_retail": ("Individual / Retail", "أفراد / تجزئة"),
    "segment": ("Segment", "الشريحة"),
    "power": ("Separability", "قوة الفصل"),
    "direction": ("Direction", "الاتجاه"),
    "bulk": ("Bulk", "جملة"),
    "retail": ("Retail", "تجزئة"),
    "orders": ("Orders", "الطلبات"),
    "share_pct": ("Share (%)", "النسبة (%)"),
    "pow_high": ("High", "عالية"),
    "pow_moderate": ("Moderate", "متوسطة"),
    "pow_low": ("Low", "منخفضة"),
    "basis_pct": ("% of orders", "% من الطلبات"),
    "basis_median": ("Median", "الوسيط"),
    "loading_data": ("Loading data…", "جاري تحميل البيانات…"),
    "loading_model": ("Loading model…", "جاري تحميل النموذج…"),
    "src_real": ("Online Retail dataset", "بيانات Online Retail"),
    "src_synth": ("Synthetic demo data", "بيانات تجريبية مولّدة"),
    "model_loaded": ("Loaded {path}", "تم تحميل {path}"),
    "model_trained": ("Trained on the fly (no .joblib found)", "تم التدريب مباشرة (لا يوجد ملف .joblib)"),

    # --- features ----------------------------------------------------------
    "f_Quantity": ("Quantity", "الكمية (Quantity)"),
    "f_UnitPrice": ("UnitPrice", "سعر الوحدة (UnitPrice)"),
    "f_Is_UK": ("Is_UK", "طلب من بريطانيا (Is_UK)"),
    "f_Is_Guest": ("Is_Guest", "شراء كزائر (Is_Guest)"),
    "f_Hour": ("Hour", "الساعة (Hour)"),
    "f_DayOfWeek": ("DayOfWeek", "يوم الأسبوع (DayOfWeek)"),
    "f_Month": ("Month", "الشهر (Month)"),
    "f_Unique_SKUs": ("Unique_SKUs", "عدد الأصناف (Unique_SKUs)"),
    "f_TotalRevenue": ("TotalRevenue", "إجمالي الإيراد (TotalRevenue)"),
    "fd_Quantity": ("Total units on the invoice. Bulk orders are defined by large volumes (≥ 50 units).",
                    "إجمالي الوحدات في الفاتورة. طلبات الجملة تُعرَّف بالكميات الكبيرة (≥ 50 وحدة)."),
    "fd_Unique_SKUs": ("Distinct product codes in the basket. Wholesale buyers restock a wide assortment.",
                       "عدد أكواد المنتجات المختلفة في السلة. تجار الجملة بيشتروا تشكيلة واسعة."),
    "fd_UnitPrice": ("Average unit price (£) across the invoice lines.", "متوسط سعر الوحدة (£) على بنود الفاتورة."),
    "fd_Hour": ("Hour of the day the invoice was created (0-23).", "ساعة إنشاء الفاتورة خلال اليوم (0-23)."),
    "fd_DayOfWeek": ("Day of the week (0 = Monday … 6 = Sunday).", "يوم الأسبوع (0 = الإثنين … 6 = الأحد)."),
    "fd_Month": ("Calendar month of the order (captures Q4 restocking seasonality).",
                 "شهر الطلب (يوضّح موسمية إعادة التخزين في الربع الرابع)."),
    "fd_Is_UK": ("1 if the order ships to the United Kingdom, otherwise 0.", "1 لو الطلب مشحون للمملكة المتحدة، غير كده 0."),
    "fd_Is_Guest": ("1 if the order has no registered customer ID (guest checkout).",
                    "1 لو الطلب من غير رقم عميل مسجّل (شراء كزائر)."),

    # --- metric guide ------------------------------------------------------
    "mg_metric": ("Metric", "المقياس"),
    "mg_tells": ("What it tells you", "ماذا يخبرك"),
    "mg_scale": ("Scale", "المقياس الرقمي"),
    "mg_ks": ("Largest gap between the two cumulative curves (Bulk vs Retail)",
              "أكبر فجوة بين المنحنيين التراكميين (جملة مقابل تجزئة)"),
    "mg_ks_s": ("0 = identical → 1 = fully separated", "0 = متطابقان ← 1 = منفصلان تمامًا"),
    "mg_js": ("Symmetric, bounded difference between the two distributions", "فرق متماثل ومحدود بين التوزيعين"),
    "mg_kl": ("Asymmetric, *unbounded* difference (can look huge when ranges barely overlap)",
              "فرق غير متماثل و*غير محدود* (قد يبدو ضخمًا لما النطاقات بالكاد تتداخل)"),
    "mg_d": ("Difference in means in standard-deviation units (sign = which segment is higher)",
             "فرق المتوسطات بوحدات الانحراف المعياري (الإشارة = أي شريحة أعلى)"),
    "mg_d_s": ("|d| 0.2 small · 0.5 medium · 0.8 large", "|d| ‏0.2 صغير · 0.5 متوسط · 0.8 كبير"),
    "mg_mi": ("How much knowing the feature reduces uncertainty about the segment",
              "كام معرفة الخاصية بتقلل عدم اليقين عن الشريحة"),
    "mg_mi_s": ("0 → higher is better", "0 ← الأعلى أفضل"),
    "mg_p": ("With thousands of orders it is almost always ≈ 0 — **rely on effect size, not p**",
             "مع آلاف الطلبات بتكون تقريبًا ≈ 0 دائمًا — **اعتمد على حجم الأثر مش على p**"),

    # --- overview ----------------------------------------------------------
    "ov_title": ("🏪 E-Commerce Data Pipeline & Buyer Segmentation", "🏪 خط معالجة بيانات التجارة الإلكترونية وتقسيم المشترين"),
    "ov_inv": ("Total Clean Invoices", "إجمالي الفواتير النظيفة"),
    "ov_inv_d": ("Net of cancellations", "بعد استبعاد الإلغاءات"),
    "ov_ratio": ("Bulk Order Ratio", "نسبة طلبات الجملة"),
    "ov_ratio_d": ("Share of invoices", "من إجمالي الفواتير"),
    "ov_aov": ("Average Order Value", "متوسط قيمة الطلب"),
    "ov_aov_d": ("Net AOV", "صافي متوسط القيمة"),
    "ov_rev": ("Bulk Revenue Share", "حصة الجملة من الإيراد"),
    "ov_rev_d": ("Share of revenue", "من إجمالي الإيراد"),
    "ov_pie_orders": ("Share of orders", "نسبة الطلبات"),
    "ov_pie_rev": ("Share of revenue", "نسبة الإيراد"),
    "ov_pipe_card": (
        "<h3>⚙️ End-to-End Data Pipeline</h3>"
        "<p>Raw retail logs contain cancellation invoices (prefix <code>C</code>), zero-price adjustments and guest "
        "checkouts. The pipeline removes cancellations, keeps guest orders via invoice-level grouping, and engineers "
        "market and time signals (<code>Is_UK</code>, <code>Hour</code>, <code>DayOfWeek</code>, <code>Month</code>).</p>"
        "<p>A <b>Random Forest</b> then classifies each order as <b>Bulk / Wholesale (B2B)</b> or "
        "<b>Individual / Retail (B2C)</b> to support inventory allocation and targeted discounting.</p>",
        "<h3>⚙️ خط معالجة البيانات من البداية للنهاية</h3>"
        "<p>سجلات البيع الخام فيها فواتير إلغاء (بادئتها <code>C</code>) وتعديلات بسعر صفر وطلبات زوّار. "
        "الخط بيشيل الإلغاءات، وبيحتفظ بطلبات الزوّار عن طريق التجميع على مستوى الفاتورة، وبيهندس "
        "إشارات السوق والوقت (<code>Is_UK</code> و<code>Hour</code> و<code>DayOfWeek</code> و<code>Month</code>).</p>"
        "<p>بعد كده <b>Random Forest</b> بيصنّف كل طلب إلى <b>جملة / تجار (B2B)</b> أو "
        "<b>أفراد / تجزئة (B2C)</b> لدعم توزيع المخزون وتخصيص الخصومات.</p>"),
    "ov_bulk_card": (
        '<h3 style="color:#39FF14;">🟢 Bulk / Wholesale (B2B signal)</h3><ul>'
        "<li><b>High volume:</b> ≥ 50 units or ≥ £500 basket value.</li>"
        "<li><b>Wide assortment:</b> dozens of distinct SKUs per invoice.</li>"
        "<li><b>Business hours:</b> weekdays 09:00–17:00, peaking in Q4.</li></ul>",
        '<h3 style="color:#39FF14;">🟢 جملة / تجار (إشارة B2B)</h3><ul>'
        "<li><b>كمية كبيرة:</b> ≥ 50 وحدة أو سلة ≥ £500.</li>"
        "<li><b>تشكيلة واسعة:</b> عشرات الأصناف المختلفة في الفاتورة.</li>"
        "<li><b>ساعات العمل:</b> أيام الأسبوع 09:00–17:00 مع ذروة في الربع الرابع.</li></ul>"),
    "ov_retail_card": (
        '<h3 style="color:#00C6FF;">🔵 Individual / Retail (B2C segment)</h3><ul>'
        "<li><b>Small baskets:</b> only a few units per checkout.</li>"
        "<li><b>More guest checkouts</b> (<code>Is_Guest = 1</code>).</li>"
        "<li><b>Evenings &amp; weekends</b> with strong UK concentration.</li></ul>",
        '<h3 style="color:#00C6FF;">🔵 أفراد / تجزئة (شريحة B2C)</h3><ul>'
        "<li><b>سلات صغيرة:</b> وحدات قليلة في كل عملية شراء.</li>"
        "<li><b>شراء كزائر أكتر</b> (<code>Is_Guest = 1</code>).</li>"
        "<li><b>المساء وعطلة نهاية الأسبوع</b> مع تركّز كبير في بريطانيا.</li></ul>"),

    # --- analytics page ----------------------------------------------------
    "an_title": ("📊 Strategic Analytics & Feature Separability", "📊 التحليلات الاستراتيجية وقدرة الخصائص على الفصل"),
    "an_scope": ("Feature scope", "نطاق الخصائص"),
    "an_scope_core": ("Core features (4)", "الخصائص الأساسية (4)"),
    "an_scope_all": ("All model features (8)", "كل خصائص النموذج (8)"),
    "tab_sep": ("📐 Separability", "📐 قدرة الفصل"),
    "tab_rel": ("🔗 Relationships", "🔗 العلاقات"),
    "tab_dist": ("📈 Distributions", "📈 التوزيعات"),
    "tab_time": ("🕒 Time & Market", "🕒 الوقت والسوق"),
    "tab_model": ("🧠 Model", "🧠 النموذج"),

    # separability tab
    "sep_how": ("ℹ️ How to read these metrics", "ℹ️ إزاي تقرأ المقاييس دي"),
    "sep_takeaways": ("### 💡 Key takeaways", "### 💡 أهم الاستنتاجات"),
    "tk_top": ("**{feat}** is the strongest separator (KS = {ks:.2f}): {basis} is **{b:,.1f}** for Bulk vs **{r:,.1f}** for Retail.",
               "**{feat}** هي أقوى خاصية فاصلة (KS = {ks:.2f}): {basis} = **{b:,.1f}** للجملة مقابل **{r:,.1f}** للتجزئة."),
    "tk_high": ("**{n} of {total}** features have high separability (KS > {thr}); the rest add little on their own.",
                "**{n} من {total}** خصائص لها قدرة فصل عالية (KS > {thr})؛ والباقي إضافته قليلة لوحده."),
    "tk_low": ("**{feat}** is the weakest signal (KS = {ks:.2f}) — the two segments behave almost alike on it.",
               "**{feat}** هي أضعف إشارة (KS = {ks:.2f}) — الشريحتان متشابهتان تقريبًا فيها."),
    "tk_leak": ("⚠️ **Quantity has KS ≈ 1.0 by construction**: `Is_Bulk` is defined as Quantity ≥ 50 *or* revenue ≥ £500, "
                "so the label leaks into this feature. Treat the model's accuracy as an upper bound; "
                "retrain without `Quantity` (and revenue-derived signals) to test real-world generalisation.",
                "⚠️ **الكمية فيها KS ≈ 1.0 بحكم التعريف**: `Is_Bulk` معرّف بأنه الكمية ≥ 50 *أو* الإيراد ≥ £500، "
                "يعني التصنيف متسرّب جوه الخاصية دي. اعتبر دقة النموذج حدًّا أعلى؛ "
                "أعد التدريب من غير `Quantity` (والإشارات المشتقة من الإيراد) علشان تختبر التعميم في الواقع."),
    "col_rank": ("Rank", "الترتيب"),
    "col_feature": ("Feature", "الخاصية"),
    "col_basis": ("Basis", "الأساس"),
    "col_basis_help": ("Median for numeric features, % of orders for binary flags",
                       "الوسيط للخصائص الرقمية، و% من الطلبات للخصائص الثنائية"),
    "col_ratio": ("Ratio (B/R)", "النسبة (ج/ت)"),
    "col_cohen": ("Cohen's d", "Cohen's d"),
    "col_mi": ("Mutual info", "المعلومات المتبادلة"),
    "col_p": ("p-value", "قيمة p"),
    "download_csv": ("⬇️ Download table (CSV)", "⬇️ تحميل الجدول (CSV)"),
    "ks_title": ("Separability ranking (KS statistic)", "ترتيب قدرة الفصل (إحصائية KS)"),
    "ks_axis": ("KS statistic (0 = identical, 1 = fully separated)", "إحصائية KS (0 = متطابق، 1 = منفصل تمامًا)"),
    "hm_title": ("Do all metrics agree? (each column scaled 0–1)", "هل كل المقاييس متفقة؟ (كل عمود مُعاير 0–1)"),
    "ef_title": ("Effect size & direction (Cohen's d)", "حجم الأثر والاتجاه (Cohen's d)"),
    "ef_axis": ("← Retail higher | Bulk higher →  (dotted = large effect ±0.8)",
                "← التجزئة أعلى | الجملة أعلى →  (المنقّط = أثر كبير ±0.8)"),
    "dir_bulk_hi": ("Higher in Bulk", "أعلى في الجملة"),
    "dir_retail_hi": ("Higher in Retail", "أعلى في التجزئة"),
    "dir_bulk_hi2": ("Bulk higher", "الجملة أعلى"),
    "dir_retail_hi2": ("Retail higher", "التجزئة أعلى"),
    "ratio_title": ("How many times bigger is the Bulk mean? (log scale)", "كام مرة متوسط الجملة أكبر؟ (مقياس لوغاريتمي)"),
    "mean_iqr": ("📋 Mean & IQR comparison", "📋 مقارنة المتوسط والمدى الربيعي (IQR)"),
    "col_bulk_mean": ("Bulk Mean", "متوسط الجملة"),
    "col_retail_mean": ("Retail Mean", "متوسط التجزئة"),
    "col_bulk_iqr": ("Bulk IQR", "IQR الجملة"),
    "col_retail_iqr": ("Retail IQR", "IQR التجزئة"),

    # relationships tab
    "rel_method": ("Correlation method", "طريقة الارتباط"),
    "rel_method_help": ("Spearman is more robust to the heavy tails in Quantity and UnitPrice.",
                        "Spearman أكثر تحمّلًا للقيم المتطرفة في الكمية والسعر."),
    "pearson": ("Pearson", "بيرسون"),
    "spearman": ("Spearman", "سبيرمان"),
    "corr_title": ("Feature correlation", "ارتباط الخصائص"),
    "corr_target": ("Correlation with the Bulk label", "الارتباط مع تصنيف الجملة"),
    "push_bulk": ("Pushes toward Bulk", "يدفع نحو الجملة"),
    "push_retail": ("Pushes toward Retail", "يدفع نحو التجزئة"),
    "col_corr": ("Correlation", "الارتباط"),
    "col_fa": ("Feature A", "الخاصية أ"),
    "col_fb": ("Feature B", "الخاصية ب"),
    "col_strength": ("Strength", "القوة"),
    "weak": ("Weak", "ضعيف"),
    "moderate": ("Moderate", "متوسط"),
    "strong": ("Strong", "قوي"),
    "top_pairs": ("**Most correlated feature pairs** (watch for redundancy)", "**أكثر أزواج الخصائص ارتباطًا** (انتبه للتكرار)"),
    "explorer": ("### 🔍 Relationship explorer", "### 🔍 مستكشف العلاقات"),
    "x_axis": ("X axis", "المحور السيني"),
    "y_axis": ("Y axis", "المحور الصادي"),
    "log_scale": ("Log scale", "مقياس لوغاريتمي"),
    "parallel_title": ("Parallel coordinates — each line is one order (green = Bulk, blue = Retail)",
                       "الإحداثيات المتوازية — كل خط يمثل طلبًا (أخضر = جملة، أزرق = تجزئة)"),
    "scatter_matrix": ("🧩 Pairwise scatter matrix", "🧩 مصفوفة الانتشار الثنائية"),
    "scatter_matrix_title": ("Pairwise scatter matrix", "مصفوفة الانتشار الثنائية"),
    "scatter_err": ("Unable to display scatter matrix: {err}", "تعذّر عرض مصفوفة الانتشار: {err}"),
    "vs": ("{x} vs {y}", "{x} مقابل {y}"),

    # distributions tab
    "dist_feature": ("Feature", "الخاصية"),
    "dist_view": ("View", "طريقة العرض"),
    "view_violin": ("Violin", "كمان (Violin)"),
    "view_kde": ("KDE", "كثافة (KDE)"),
    "view_hist": ("Histogram", "مدرج تكراري"),
    "ks_stat": ("KS statistic", "إحصائية KS"),
    "rf_imp": ("RF importance", "أهمية RF"),
    "dist_share_title": ("Share of orders with {feat} = 1", "نسبة الطلبات التي {feat} = 1"),
    "dist_violin": ("{feat} by segment (violin)", "{feat} حسب الشريحة (Violin)"),
    "dist_kde": ("{feat} density (KDE)", "كثافة {feat} (KDE)"),
    "density": ("Density", "الكثافة"),
    "dist_hist": ("{feat} distribution", "توزيع {feat}"),
    "ecdf_max": ("Max gap (KS)", "أكبر فجوة (KS)"),
    "ecdf_title": ("Cumulative curves — the KS gap for {feat}", "المنحنيات التراكمية — فجوة KS للخاصية {feat}"),
    "ecdf_y": ("Cumulative share of orders", "النسبة التراكمية للطلبات"),
    "dist_cap": ("Right: the dashed white line is the largest vertical gap between the two cumulative curves — "
                 "that gap *is* the KS statistic.",
                 "الرسم الأيمن: الخط الأبيض المتقطع هو أكبر فجوة رأسية بين المنحنيين التراكميين — "
                 "وهذه الفجوة هي إحصائية KS نفسها."),

    # time & market tab
    "hm_bulk_title": ("When do Bulk orders happen? (% of orders that are Bulk)", "إمتى بتحصل طلبات الجملة؟ (% الطلبات اللي هي جملة)"),
    "hour_of_day": ("Hour of day", "ساعة اليوم"),
    "day": ("Day", "اليوم"),
    "pct_bulk": ("% Bulk", "% جملة"),
    "hover_hm": ("Day: %{y}<br>Hour: %{x}:00<br>Bulk share: %{z:.1f}%<extra></extra>",
                 "اليوم: %{y}<br>الساعة: %{x}:00<br>نسبة الجملة: %{z:.1f}%<extra></extra>"),
    "weekly_title": ("Weekly pattern (% of each segment's orders)", "النمط الأسبوعي (% من طلبات كل شريحة)"),
    "monthly_title": ("Monthly seasonality (% of each segment's orders)", "الموسمية الشهرية (% من طلبات كل شريحة)"),
    "flag_uk": ("UK orders", "طلبات بريطانيا"),
    "flag_guest": ("Guest checkouts", "شراء كزائر"),
    "flag": ("Flag", "المؤشر"),
    "flags_title": ("Market & checkout profile (%)", "ملف السوق وطريقة الدفع (%)"),
    "hourly_sel": ("Hourly mean of", "المتوسط بالساعة لـ"),
    "hourly_title": ("Hourly mean of {feat}", "المتوسط بالساعة لـ {feat}"),

    # model tab
    "imp_col": ("Importance (%)", "الأهمية (%)"),
    "imp_title": ("Random Forest feature importance", "أهمية الخصائص في Random Forest"),
    "stat_vs_model": ("Statistics vs. model: do they agree?", "الإحصاء مقابل النموذج: هل يتفقان؟"),
    "model_cap": ("Top-right = strong on both the statistical test and in the model. "
                  "High KS with low importance usually means the feature is redundant with another one.",
                  "أعلى اليمين = قوي إحصائيًا وفي النموذج معًا. "
                  "KS عالية مع أهمية منخفضة غالبًا معناها إن الخاصية مكررة مع خاصية تانية."),
    "pca_spin": ("Computing decision surface…", "جاري حساب سطح القرار…"),
    "pca_warn": ("Not enough valid rows for a PCA projection.", "عدد الصفوف الصالحة غير كافٍ لإسقاط PCA."),
    "pca_title": ("PCA projection with model decision surface (explained variance {v0:.1f}% + {v1:.1f}%)",
                  "إسقاط PCA مع سطح قرار النموذج (التباين المفسَّر {v0:.1f}% + {v1:.1f}%)"),
    "pca_cap": ("2-D PCA projection of the 8-D feature space; the background is an approximation of the model's "
                "P(Bulk) obtained by mapping the 2-D grid back to feature space.",
                "إسقاط PCA ثنائي الأبعاد لفضاء الخصائص ذي الثمانية أبعاد؛ الخلفية تقريب لاحتمال الجملة P(Bulk) "
                "في النموذج، ومحسوب بإعادة تحويل الشبكة ثنائية الأبعاد إلى فضاء الخصائص."),

    # --- classifier page ---------------------------------------------------
    "cl_title": ("🤖 Real-Time Order Classifier", "🤖 مصنّف الطلبات اللحظي"),
    "cl_intro": ("<p>Enter the basket parameters to classify an incoming checkout as "
                 "<b>Bulk / Wholesale (B2B)</b> or <b>Individual / Retail (B2C)</b>.</p>",
                 "<p>أدخل بيانات السلة لتصنيف عملية الشراء الواردة إلى "
                 "<b>جملة / تجار (B2B)</b> أو <b>أفراد / تجزئة (B2C)</b>.</p>"),
    "cl_qty": ("Total Quantity (units)", "إجمالي الكمية (وحدات)"),
    "cl_price": ("Average Unit Price (£)", "متوسط سعر الوحدة (£)"),
    "cl_skus": ("Unique SKUs in basket", "عدد الأصناف المختلفة في السلة"),
    "cl_market": ("Market (Is_UK)", "السوق (Is_UK)"),
    "cl_uk": ("United Kingdom (1)", "المملكة المتحدة (1)"),
    "cl_intl": ("International (0)", "دولي (0)"),
    "cl_checkout": ("Checkout type (Is_Guest)", "نوع الشراء (Is_Guest)"),
    "cl_reg": ("Registered account (0)", "حساب مسجّل (0)"),
    "cl_guest": ("Guest checkout (1)", "شراء كزائر (1)"),
    "cl_hour": ("Order hour (24h)", "ساعة الطلب (24 ساعة)"),
    "cl_day": ("Day of week", "يوم الأسبوع"),
    "cl_month": ("Month", "الشهر"),
    "cl_submit": ("Analyze Order", "تحليل الطلب"),
    "cl_verdict": ("Order Classification Verdict", "نتيجة تصنيف الطلب"),
    "cl_bulk_title": ("BULK / WHOLESALE (B2B)", "جملة / تجار (B2B)"),
    "cl_retail_title": ("INDIVIDUAL / RETAIL (B2C)", "أفراد / تجزئة (B2C)"),
    "cl_value": ("Estimated basket value", "القيمة التقديرية للسلة"),
    "cl_p_bulk": ("Bulk probability", "احتمال الجملة"),
    "cl_p_retail": ("Retail probability", "احتمال التجزئة"),
    "cl_rule": ("Business rule (≥ 50 units or ≥ £500) says **{rule}** — {verdict} the model.",
                "قاعدة العمل (≥ 50 وحدة أو ≥ £500) تقول **{rule}** — {verdict} النموذج."),
    "cl_consistent": ("consistent with", "متوافقة مع"),
    "cl_differs": ("differs from", "تختلف عن"),
    "cl_gauge": ("Prediction Confidence", "ثقة التنبؤ"),

    # --- sidebar / nav -----------------------------------------------------
    "nav_title": ("🛒 E-Commerce BI", "🛒 ذكاء أعمال التجارة الإلكترونية"),
    "nav_caption": ("Strategic retail analytics & B2B/B2C order classification",
                    "تحليلات تجزئة استراتيجية وتصنيف طلبات B2B/B2C"),
    "nav_label": ("Navigation", "التنقل"),
    "nav_overview": ("🏪 Executive Overview", "🏪 نظرة تنفيذية"),
    "nav_analytics": ("📊 Strategic Analytics", "📊 التحليلات الاستراتيجية"),
    "nav_classifier": ("🤖 Order Classifier", "🤖 مصنّف الطلبات"),
    "side_data": ("**Data:** {src} · {n:,} invoices", "**البيانات:** {src} · {n:,} فاتورة"),
    "side_model": ("**Model:** {src}", "**النموذج:** {src}"),
}

# ════════════════════════════════════════════════════════════════════════════
# 2. CONFIG & CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
LANG = st.session_state.get("lang", "en")
IS_AR = LANG == "ar"


def t(key: str, **kw) -> str:
    """Translate `key` into the active language (optionally .format(**kw))."""
    s = TR[key][1 if IS_AR else 0]
    return s.format(**kw) if kw else s


st.set_page_config(
    page_title=t("page_title"),
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Internal (data) keys stay English; the *_T / display objects below are translated.
BULK, RETAIL = t("seg_bulk"), t("seg_retail")
SEG_COLORS = {BULK: "#39FF14", RETAIL: "#00C6FF"}
SEG_ORDER = {"Segment": [BULK, RETAIL]}

FEATURE_COLS = ["Quantity", "UnitPrice", "Is_UK", "Is_Guest", "Hour", "DayOfWeek", "Month", "Unique_SKUs"]
CORE_FEATURES = ["Quantity", "Unique_SKUs", "UnitPrice", "Hour"]
BINARY_FEATURES = {"Is_UK", "Is_Guest"}
DISCRETE_FEATURES = BINARY_FEATURES | {"Hour", "DayOfWeek", "Month"}
HEAVY_TAIL = {"Quantity", "UnitPrice", "Unique_SKUs", "TotalRevenue"}

MODEL_PATH = "best_rf_pipeline.joblib"
KS_HIGH, KS_MODERATE = 0.35, 0.20

if IS_AR:
    DAY_NAMES = {0: "الإثنين", 1: "الثلاثاء", 2: "الأربعاء", 3: "الخميس", 4: "الجمعة", 5: "السبت", 6: "الأحد"}
    MONTH_NAMES = {1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل", 5: "مايو", 6: "يونيو",
                   7: "يوليو", 8: "أغسطس", 9: "سبتمبر", 10: "أكتوبر", 11: "نوفمبر", 12: "ديسمبر"}
    DAY_FULL = list(DAY_NAMES.values())
else:
    DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    DAY_FULL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Separability levels: internal key (English) -> display name
PW = {"High": t("pow_high"), "Moderate": t("pow_moderate"), "Low": t("pow_low")}
POWER_COLORS = {PW["High"]: "#39FF14", PW["Moderate"]: "#FFC857", PW["Low"]: "#8892A6"}
POWER_LABEL = {"High": "🟢 " + PW["High"], "Moderate": "🟡 " + PW["Moderate"], "Low": "⚪ " + PW["Low"]}
BASIS = {"% of orders": t("basis_pct"), "Median": t("basis_median")}
SHARE = t("share_pct")

# Legend titles that Plotly Express derives from column names
LEGEND_TITLES = {"Segment": t("segment"), "Power": t("power"), "Direction": ""}


def flabel(feat: str) -> str:
    return t(f"f_{feat}")


def metric_guide() -> str:
    rows = [
        ("KS statistic" if not IS_AR else "إحصائية KS", t("mg_ks"), t("mg_ks_s")),
        ("JS divergence" if not IS_AR else "تباعد JS", t("mg_js"), "0 → 1"),
        ("KL divergence" if not IS_AR else "تباعد KL", t("mg_kl"), "0 → ∞"),
        ("Cohen's d", t("mg_d"), t("mg_d_s").replace("|d|", "\\|d\\|")),
        ("Mutual info" if not IS_AR else "المعلومات المتبادلة", t("mg_mi"), t("mg_mi_s")),
        ("p-value" if not IS_AR else "قيمة p", t("mg_p"), "—"),
    ]
    out = [f"| {t('mg_metric')} | {t('mg_tells')} | {t('mg_scale')} |", "|---|---|---|"]
    out += [f"| **{m}** | {d} | {s} |" for m, d, s in rows]
    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════════
# 3. STYLING HELPERS
# ════════════════════════════════════════════════════════════════════════════
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&w=2048&q=80");
            background-size: cover; background-position: center; background-attachment: fixed;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stCaptionContainer"] { color: #FFFFFF; }
        [data-testid="stSidebar"] {
            background-color: rgba(0,0,0,0.85) !important;
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255,255,255,0.15);
        }
        .glass-container, [data-testid="stMetric"] {
            background-color: rgba(0,0,0,0.75);
            backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 14px; padding: 20px 24px; margin-bottom: 16px;
        }
        [data-testid="stMetric"] { padding: 14px 18px; }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color: #FFFFFF !important; }
        .glass-container h3 { margin-top: 0; }
        .stTabs [data-baseweb="tab"] { font-weight: 600; }
        .stDataFrame { border-radius: 10px; }
        div.stButton > button, div.stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
            color: #FFFFFF; border: none; border-radius: 8px;
            padding: 0.6rem 1.5rem; font-weight: bold; transition: all 0.25s ease;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            transform: scale(1.02); box-shadow: 0 0 15px rgba(0,198,255,0.6);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if IS_AR:
        # Right-to-left layout + Arabic font. Charts, tables, sliders and
        # number inputs stay left-to-right so they keep rendering correctly.
        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
            .stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
            .stApp button, .stApp [data-testid="stMarkdownContainer"], .glass-container,
            [data-testid="stMetric"], .stTabs [data-baseweb="tab"] { font-family: 'Cairo', sans-serif; }
            [data-testid="stMain"] .block-container,
            [data-testid="stSidebarContent"] { direction: rtl; text-align: right; }
            [data-testid="stPlotlyChart"], [data-testid="stDataFrame"],
            [data-testid="stSlider"], [data-testid="stNumberInput"] input { direction: ltr; text-align: left; }
            [data-testid="stSlider"] label, [data-testid="stNumberInput"] label { direction: rtl; text-align: right; display: block; }
            [data-testid="stSidebar"] { border-right: none; border-left: 1px solid rgba(255,255,255,0.15); }
            .glass-container ul { padding-right: 1.2rem; padding-left: 0; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def card(html: str, border: str | None = None) -> None:
    style = f' style="border: 2px solid {border};"' if border else ""
    st.markdown(f'<div class="glass-container"{style}>{html}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    font = dict(color="#FFFFFF")
    if IS_AR:
        font["family"] = "Cairo, Segoe UI, Tahoma, sans-serif"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=font,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.2)", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.25)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.12)", zerolinecolor="rgba(255,255,255,0.25)")
    # translate legend titles that Plotly Express generated from column names
    cur = fig.layout.legend.title.text if fig.layout.legend and fig.layout.legend.title else None
    if cur in LEGEND_TITLES:
        fig.update_layout(legend_title_text=LEGEND_TITLES[cur])
    if height:
        fig.update_layout(height=height)
    return fig


def show(fig: go.Figure, height: int | None = None) -> None:
    st.plotly_chart(style_fig(fig, height), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# 4. DATA & MODEL LOADING   (language independent -> safe to cache)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading data…")
def load_ecommerce_data() -> tuple[pd.DataFrame, str]:
    """Returns (invoice-level dataframe, data source key: 'real' | 'synthetic')."""
    try:
        try:
            raw = pd.read_excel("Online Retail.xlsx", dtype={"InvoiceNo": str, "StockCode": str})
        except FileNotFoundError:
            raw = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1", dtype={"InvoiceNo": str, "StockCode": str})

        raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
        raw["UnitPrice"] = pd.to_numeric(raw["UnitPrice"], errors="coerce")
        raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
        raw = raw.dropna(subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"])

        inv_customer = raw.groupby("InvoiceNo")["CustomerID"].transform("first")
        raw["CustomerID"] = raw["CustomerID"].fillna(inv_customer)
        raw["Is_Guest"] = raw["CustomerID"].isna().astype(int)

        sales = raw[
            (raw["UnitPrice"] > 0)
            & (~raw["InvoiceNo"].str.startswith("C", na=False))
            & (raw["Quantity"] > 0)
        ].copy()
        sales["LineRevenue"] = sales["Quantity"] * sales["UnitPrice"]
        sales["Is_UK"] = (sales["Country"] == "United Kingdom").astype(int)
        sales["Hour"] = sales["InvoiceDate"].dt.hour
        sales["DayOfWeek"] = sales["InvoiceDate"].dt.dayofweek
        sales["Month"] = sales["InvoiceDate"].dt.month

        df = sales.groupby("InvoiceNo").agg(
            Quantity=("Quantity", "sum"),
            UnitPrice=("UnitPrice", "mean"),
            Is_UK=("Is_UK", "first"),
            Is_Guest=("Is_Guest", "first"),
            Hour=("Hour", "first"),
            DayOfWeek=("DayOfWeek", "first"),
            Month=("Month", "first"),
            Unique_SKUs=("StockCode", "nunique"),
            TotalRevenue=("LineRevenue", "sum"),
        ).reset_index()
        df["Is_Bulk"] = np.where((df["Quantity"] >= 50) | (df["TotalRevenue"] >= 500), 1, 0)
        source = "real"

    except FileNotFoundError:
        rng = np.random.default_rng(42)
        n_retail, n_bulk = 14500, 4000
        retail = pd.DataFrame({
            "Quantity": rng.integers(1, 35, n_retail),
            "UnitPrice": rng.uniform(1.5, 25.0, n_retail),
            "Is_UK": rng.choice([1, 0], size=n_retail, p=[0.90, 0.10]),
            "Is_Guest": rng.choice([0, 1], size=n_retail, p=[0.85, 0.15]),
            "Hour": rng.integers(8, 20, n_retail),
            "DayOfWeek": rng.integers(0, 7, n_retail),
            "Month": rng.choice(range(1, 13), size=n_retail),
            "Unique_SKUs": rng.integers(1, 15, n_retail),
            "Is_Bulk": 0,
        })
        bulk = pd.DataFrame({
            "Quantity": rng.integers(50, 600, n_bulk),
            "UnitPrice": rng.uniform(2.0, 45.0, n_bulk),
            "Is_UK": rng.choice([1, 0], size=n_bulk, p=[0.75, 0.25]),
            "Is_Guest": rng.choice([0, 1], size=n_bulk, p=[0.95, 0.05]),
            "Hour": rng.integers(9, 17, n_bulk),
            "DayOfWeek": rng.integers(0, 5, n_bulk),
            "Month": rng.choice([9, 10, 11, 12], size=n_bulk),
            "Unique_SKUs": rng.integers(10, 80, n_bulk),
            "Is_Bulk": 1,
        })
        df = pd.concat([retail, bulk], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        df["TotalRevenue"] = df["Quantity"] * df["UnitPrice"]
        source = "synthetic"

    return df, source


@st.cache_resource(show_spinner="Loading model…")
def load_model(_df: pd.DataFrame):
    """Returns (model, source key: 'loaded' | 'trained')."""
    try:
        return joblib.load(MODEL_PATH), "loaded"
    except FileNotFoundError:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ])
        pipe.fit(_df[FEATURE_COLS], _df["Is_Bulk"])
        return pipe, "trained"


def get_feature_importances(model, names: list[str]) -> dict[str, float]:
    steps = list(model.named_steps.values())[::-1] if hasattr(model, "named_steps") else [model]
    for step in steps:
        if hasattr(step, "feature_importances_"):
            return dict(zip(names, step.feature_importances_))
    return {f: 1.0 / len(names) for f in names}


# ════════════════════════════════════════════════════════════════════════════
# 5. STATISTICS   (internal column names stay English -> cache is language independent)
# ════════════════════════════════════════════════════════════════════════════
def _clean(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()


def _divergences(b: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """Returns (Jensen-Shannon divergence, KL divergence B||R) on a shared 30-bin histogram."""
    both = np.concatenate([b, r])
    if both.min() == both.max():
        return 0.0, 0.0
    bins = np.histogram_bin_edges(both, bins=30)
    p, _ = np.histogram(b, bins=bins)
    q, _ = np.histogram(r, bins=bins)
    p = (p + 1e-10) / (p + 1e-10).sum()
    q = (q + 1e-10) / (q + 1e-10).sum()
    return float(jensenshannon(p, q, base=2.0) ** 2), float(entropy(p, q))


def _power(ks: float) -> str:
    return "High" if ks > KS_HIGH else ("Moderate" if ks > KS_MODERATE else "Low")


@st.cache_data(show_spinner=False)
def compute_separability(_df: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    bulk, retail = _df[_df["Is_Bulk"] == 1], _df[_df["Is_Bulk"] == 0]

    sample = _df.sample(min(20000, len(_df)), random_state=42)
    X = sample[list(features)].apply(pd.to_numeric, errors="coerce").fillna(0)
    mi = mutual_info_classif(
        X, sample["Is_Bulk"], discrete_features=[f in DISCRETE_FEATURES for f in features], random_state=42
    )
    mi_map = dict(zip(features, mi))

    rows = []
    for feat in features:
        b, r = _clean(bulk[feat]), _clean(retail[feat])
        if len(b) < 2 or len(r) < 2:
            continue
        ks, p = ks_2samp(b, r)
        js, kl = _divergences(b, r)
        is_bin = feat in BINARY_FEATURES

        typ_b = float(b.mean() * 100) if is_bin else float(np.median(b))
        typ_r = float(r.mean() * 100) if is_bin else float(np.median(r))
        pooled = np.sqrt(((len(b) - 1) * b.var(ddof=1) + (len(r) - 1) * r.var(ddof=1)) / (len(b) + len(r) - 2))
        d = float((b.mean() - r.mean()) / pooled) if pooled > 0 else 0.0
        ratio = float(b.mean() / r.mean()) if abs(r.mean()) > 1e-9 else np.nan
        rows.append({
            "Feature": feat,
            "Power": _power(ks),
            "Basis": "% of orders" if is_bin else "Median",
            "Bulk": round(typ_b, 2),
            "Retail": round(typ_r, 2),
            "Ratio (B/R)": round(ratio, 2) if pd.notna(ratio) else np.nan,
            "KS": round(float(ks), 4),
            "JS": round(js, 4),
            "KL": round(kl, 4),
            "Cohen d": round(d, 3),
            "Mutual Info": round(float(mi_map[feat]), 4),
            "p-value": max(float(p), 1e-300),
        })

    cols = ["Rank", "Feature", "Power", "Basis", "Bulk", "Retail", "Ratio (B/R)",
            "KS", "JS", "KL", "Cohen d", "Mutual Info", "p-value"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows).sort_values("KS", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", out.index + 1)
    return out


@st.cache_data(show_spinner=False)
def compute_correlations(_df: pd.DataFrame, method: str) -> pd.DataFrame:
    cols = FEATURE_COLS + (["TotalRevenue"] if "TotalRevenue" in _df.columns else []) + ["Is_Bulk"]
    return _df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method).round(2)


def top_feature_pairs(corr: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    feats = [c for c in corr.columns if c not in ("Is_Bulk", "TotalRevenue")]
    rows = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:]]
    pairs = pd.DataFrame(rows, columns=["Feature A", "Feature B", "Correlation"])
    pairs["Strength"] = pd.cut(pairs["Correlation"].abs(), [0, 0.3, 0.5, 1.0001],
                               labels=[t("weak"), t("moderate"), t("strong")], include_lowest=True)
    pairs = pairs.reindex(pairs["Correlation"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)
    return pairs.rename(columns={"Feature A": t("col_fa"), "Feature B": t("col_fb"),
                                 "Correlation": t("col_corr"), "Strength": t("col_strength")})


def segment_comparison_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        b = pd.Series(_clean(df.loc[df["Is_Bulk"] == 1, feat]))
        r = pd.Series(_clean(df.loc[df["Is_Bulk"] == 0, feat]))
        if b.empty or r.empty:
            continue
        rows.append({
            t("col_feature"): feat,
            t("col_bulk_mean"): round(b.mean(), 2), t("col_retail_mean"): round(r.mean(), 2),
            t("col_bulk_iqr"): round(b.quantile(0.75) - b.quantile(0.25), 2),
            t("col_retail_iqr"): round(r.quantile(0.75) - r.quantile(0.25), 2),
        })
    return pd.DataFrame(rows)


def build_takeaways(sep: pd.DataFrame) -> list[str]:
    if sep.empty:
        return []
    top, low = sep.iloc[0], sep.iloc[-1]
    n_high = int((sep["Power"] == "High").sum())
    notes = [
        t("tk_top", feat=top["Feature"], ks=top["KS"], basis=BASIS[top["Basis"]], b=top["Bulk"], r=top["Retail"]),
        t("tk_high", n=n_high, total=len(sep), thr=KS_HIGH),
        t("tk_low", feat=low["Feature"], ks=low["KS"]),
    ]
    q = sep[sep["Feature"] == "Quantity"]
    if not q.empty and q["KS"].iloc[0] >= 0.98:
        notes.append(t("tk_leak"))
    return notes


# ════════════════════════════════════════════════════════════════════════════
# 6. CHART BUILDERS
# ════════════════════════════════════════════════════════════════════════════
def fig_ks_ranking(sep: pd.DataFrame) -> go.Figure:
    d = sep.sort_values("KS").copy()
    d["Power"] = d["Power"].map(PW)
    fig = px.bar(d, x="KS", y="Feature", orientation="h", color="Power",
                 color_discrete_map=POWER_COLORS,
                 category_orders={"Power": [PW["High"], PW["Moderate"], PW["Low"]]},
                 text=d["KS"].map("{:.2f}".format), title=t("ks_title"))
    fig.update_traces(textposition="outside")
    for x, label in [(KS_MODERATE, PW["Moderate"]), (KS_HIGH, PW["High"])]:
        fig.add_vline(x=x, line_dash="dot", line_color="rgba(255,255,255,0.5)",
                      annotation_text=label, annotation_position="top")
    fig.update_xaxes(range=[0, 1.1], title=t("ks_axis"))
    fig.update_yaxes(title=None)
    return fig


def fig_metric_heatmap(sep: pd.DataFrame) -> go.Figure:
    m = sep.set_index("Feature")[["KS", "JS", "Cohen d", "Mutual Info", "KL"]].copy()
    m["Cohen d"] = m["Cohen d"].abs()
    m = m.rename(columns={"Cohen d": "|Cohen d|"})
    scaled = m / m.max().replace(0, 1)
    fig = px.imshow(scaled, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis",
                    zmin=0, zmax=1, title=t("hm_title"))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(side="top")
    return fig


def fig_effect_size(sep: pd.DataFrame) -> go.Figure:
    d = sep.sort_values("Cohen d").copy()
    d["Direction"] = np.where(d["Cohen d"] >= 0, t("dir_bulk_hi"), t("dir_retail_hi"))
    fig = px.bar(d, x="Cohen d", y="Feature", orientation="h", color="Direction",
                 color_discrete_map={t("dir_bulk_hi"): SEG_COLORS[BULK], t("dir_retail_hi"): SEG_COLORS[RETAIL]},
                 title=t("ef_title"))
    for x in (-0.8, 0.8):
        fig.add_vline(x=x, line_dash="dot", line_color="rgba(255,255,255,0.4)")
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.6)")
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=t("ef_axis"))
    return fig


def fig_ratio(sep: pd.DataFrame) -> go.Figure:
    d = sep.dropna(subset=["Ratio (B/R)"]).copy()
    d = d[d["Ratio (B/R)"] > 0].sort_values("Ratio (B/R)")
    d["Direction"] = np.where(d["Ratio (B/R)"] >= 1, t("dir_bulk_hi2"), t("dir_retail_hi2"))
    fig = px.bar(d, x="Ratio (B/R)", y="Feature", orientation="h", log_x=True, color="Direction",
                 color_discrete_map={t("dir_bulk_hi2"): SEG_COLORS[BULK], t("dir_retail_hi2"): SEG_COLORS[RETAIL]},
                 text=d["Ratio (B/R)"].map("{:.2f}x".format),
                 title=t("ratio_title"))
    fig.add_vline(x=1, line_color="rgba(255,255,255,0.6)")
    fig.update_yaxes(title=None)
    return fig


def fig_distribution(df: pd.DataFrame, feat: str, kind: str) -> go.Figure:
    if feat in BINARY_FEATURES:
        tbl = (df.groupby("Segment")[feat].mean() * 100).reset_index(name=SHARE)
        return px.bar(tbl, x="Segment", y=SHARE, color="Segment", color_discrete_map=SEG_COLORS,
                      labels={"Segment": t("segment")}, title=t("dist_share_title", feat=feat))

    plot_df = df[df[feat] <= df[feat].quantile(0.97)] if feat in HEAVY_TAIL else df
    if kind == "Violin":
        return px.violin(plot_df, x="Segment", y=feat, color="Segment", box=True, points=False,
                         labels={"Segment": t("segment")},
                         color_discrete_map=SEG_COLORS, title=t("dist_violin", feat=feat))
    if kind == "KDE":
        b = _clean(plot_df.loc[plot_df["Is_Bulk"] == 1, feat])
        r = _clean(plot_df.loc[plot_df["Is_Bulk"] == 0, feat])
        try:
            if len(b) > 1 and len(r) > 1 and b.std() > 0 and r.std() > 0:
                fig = ff.create_distplot([b, r], [BULK, RETAIL], colors=[SEG_COLORS[BULK], SEG_COLORS[RETAIL]],
                                         show_hist=False, show_rug=False)
                fig.update_layout(title=t("dist_kde", feat=feat), xaxis_title=feat, yaxis_title=t("density"))
                return fig
        except Exception:
            pass
    return px.histogram(plot_df, x=feat, color="Segment", barmode="overlay", opacity=0.65, nbins=40,
                        histnorm="probability density", color_discrete_map=SEG_COLORS,
                        labels={"Segment": t("segment")}, title=t("dist_hist", feat=feat))


def fig_ecdf(df: pd.DataFrame, feat: str) -> go.Figure:
    """Cumulative curves with the maximum gap (= KS statistic) drawn explicitly."""
    b = np.sort(_clean(df.loc[df["Is_Bulk"] == 1, feat]))
    r = np.sort(_clean(df.loc[df["Is_Bulk"] == 0, feat]))
    both = np.concatenate([b, r])
    vals = np.unique(both)
    grid = vals if len(vals) <= 2000 else np.unique(np.quantile(both, np.linspace(0, 1, 2000)))
    fb = np.searchsorted(b, grid, side="right") / len(b)
    fr = np.searchsorted(r, grid, side="right") / len(r)
    gap = np.abs(fb - fr)
    i = int(gap.argmax())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=fb, mode="lines", name=BULK,
                             line=dict(color=SEG_COLORS[BULK], shape="hv")))
    fig.add_trace(go.Scatter(x=grid, y=fr, mode="lines", name=RETAIL,
                             line=dict(color=SEG_COLORS[RETAIL], shape="hv")))
    fig.add_trace(go.Scatter(x=[grid[i], grid[i]], y=[fb[i], fr[i]], mode="lines+markers", name=t("ecdf_max"),
                             line=dict(color="#FFFFFF", dash="dash")))
    fig.add_annotation(x=grid[i], y=(fb[i] + fr[i]) / 2, text=f"KS = {gap[i]:.3f}", showarrow=True,
                       ax=60, ay=0, font=dict(color="#FFFFFF"), bgcolor="rgba(0,0,0,0.6)")
    if feat in HEAVY_TAIL:
        fig.update_xaxes(range=[float(grid[0]), float(np.quantile(both, 0.99))])
    fig.update_layout(title=t("ecdf_title", feat=feat), xaxis_title=feat, yaxis_title=t("ecdf_y"))
    return fig


def fig_corr_heatmap(corr: pd.DataFrame, method: str) -> go.Figure:
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    fig = px.imshow(corr.mask(mask), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title=f"{t('corr_title')} ({t(method)})")
    return fig


def fig_target_corr(corr: pd.DataFrame) -> go.Figure:
    s = corr["Is_Bulk"].drop("Is_Bulk")
    d = s.reindex(s.abs().sort_values().index).reset_index()
    d.columns = ["Feature", "Correlation"]
    d["Direction"] = np.where(d["Correlation"] >= 0, t("push_bulk"), t("push_retail"))
    fig = px.bar(d, x="Correlation", y="Feature", orientation="h", color="Direction",
                 color_discrete_map={t("push_bulk"): SEG_COLORS[BULK], t("push_retail"): SEG_COLORS[RETAIL]},
                 labels={"Correlation": t("col_corr")}, title=t("corr_target"))
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(title=None)
    return fig


def fig_relationship(df: pd.DataFrame, x: str, y: str, log: bool) -> go.Figure:
    s = df.dropna(subset=[x, y])
    if log:
        s = s[(s[x] > 0) & (s[y] > 0)]
    s = s.sample(n=min(3000, len(s)), random_state=42)
    return px.scatter(s, x=x, y=y, color="Segment", color_discrete_map=SEG_COLORS, opacity=0.6,
                      log_x=log, log_y=log, marginal_x="box", marginal_y="box",
                      labels={"Segment": t("segment")}, title=t("vs", x=x, y=y))


def fig_parallel(df: pd.DataFrame, dims: list[str]) -> go.Figure:
    s = df.sample(n=min(1500, len(df)), random_state=42).copy()
    for c in ("Quantity", "UnitPrice", "Unique_SKUs"):
        if c in dims:
            s[c] = s[c].clip(upper=s[c].quantile(0.98))
    fig = px.parallel_coordinates(
        s, dimensions=dims, color="Is_Bulk",
        color_continuous_scale=[[0, SEG_COLORS[RETAIL]], [1, SEG_COLORS[BULK]]],
        title=t("parallel_title"),
    )
    fig.update_traces(labelfont=dict(color="#FFFFFF"), tickfont=dict(color="#FFFFFF"),
                      rangefont=dict(color="#FFFFFF"))
    fig.update_coloraxes(showscale=False)
    return fig


def share_table(df: pd.DataFrame, col: str, labels: dict) -> pd.DataFrame:
    tbl = df.groupby(["Segment", col]).size().reset_index(name="Orders")
    tbl[SHARE] = (tbl["Orders"] / tbl.groupby("Segment")["Orders"].transform("sum") * 100).round(1)
    tbl["Label"] = tbl[col].map(labels)
    return tbl.sort_values(col)


def fig_bulk_heatmap(df: pd.DataFrame) -> go.Figure:
    pv = (df.pivot_table(index="DayOfWeek", columns="Hour", values="Is_Bulk", aggfunc="mean") * 100)
    pv = pv.reindex(range(7)).rename(index=DAY_NAMES)
    calm_scale = [
        [0.0, "#24445C"],
        [0.3, "#2F7A87"],
        [0.6, "#6FB7A4"],
        [0.85, "#C9DDA8"],
        [1.0, "#F3E2A9"],
    ]
    fig = px.imshow(
        pv, aspect="auto", color_continuous_scale=calm_scale, text_auto=".0f",
        labels=dict(x=t("hour_of_day"), y=t("day"), color=t("pct_bulk")),
        title=t("hm_bulk_title"),
    )
    fig.update_traces(xgap=2, ygap=2, hovertemplate=t("hover_hm"))
    fig.update_xaxes(dtick=1, side="bottom")
    fig.update_coloraxes(colorbar=dict(ticksuffix="%", title=dict(text=t("pct_bulk"))))
    return fig


def fig_pca_boundary(df: pd.DataFrame, model) -> go.Figure | None:
    valid = df.dropna(subset=FEATURE_COLS)
    if len(valid) < 5:
        return None
    sample = valid.sample(n=min(2500, len(valid)), random_state=42).copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(sample[FEATURE_COLS])
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(Xs)
    sample["PC1"], sample["PC2"] = pcs[:, 0], pcs[:, 1]

    pad_x = max((sample["PC1"].max() - sample["PC1"].min()) * 0.08, 0.5)
    pad_y = max((sample["PC2"].max() - sample["PC2"].min()) * 0.08, 0.5)
    gx = np.linspace(sample["PC1"].min() - pad_x, sample["PC1"].max() + pad_x, 60)
    gy = np.linspace(sample["PC2"].min() - pad_y, sample["PC2"].max() + pad_y, 60)
    xx, yy = np.meshgrid(gx, gy)
    grid = scaler.inverse_transform(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    zz = np.clip(model.predict_proba(pd.DataFrame(grid, columns=FEATURE_COLS))[:, 1], 0, 1).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=gx, y=gy, z=zz, hoverinfo="skip", name="P(Bulk)",
        colorscale=[[0, "rgba(0,198,255,0.35)"], [0.5, "rgba(255,255,255,0.10)"], [1, "rgba(57,255,20,0.35)"]],
        contours=dict(start=0, end=1, size=0.1, showlines=True),
        colorbar=dict(title=dict(text="P(Bulk)", font=dict(color="#FFFFFF")), tickfont=dict(color="#FFFFFF")),
    ))
    for seg in (BULK, RETAIL):
        sub = sample[sample["Segment"] == seg]
        fig.add_trace(go.Scatter(x=sub["PC1"], y=sub["PC2"], mode="markers", name=seg,
                                 marker=dict(color=SEG_COLORS[seg], size=6, opacity=0.75,
                                             line=dict(width=0.5, color="#000000"))))
    v = pca.explained_variance_ratio_ * 100
    fig.update_layout(title=t("pca_title", v0=v[0], v1=v[1]),
                      xaxis_title=f"PC1 ({v[0]:.1f}%)", yaxis_title=f"PC2 ({v[1]:.1f}%)")
    return fig


def fig_gauge(value: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 44}},
        title={"text": t("cl_gauge"), "font": {"color": "#FFFFFF", "size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#FFFFFF", "tickfont": {"color": "#FFFFFF"}},
            "bar": {"color": color}, "bgcolor": "rgba(255,255,255,0.1)",
            "borderwidth": 2, "bordercolor": "rgba(255,255,255,0.3)",
            "steps": [{"range": [0, 50], "color": "rgba(255,255,255,0.05)"},
                      {"range": [50, 80], "color": "rgba(255,255,255,0.12)"},
                      {"range": [80, 100], "color": "rgba(255,255,255,0.20)"}],
        },
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 7. PAGES
# ════════════════════════════════════════════════════════════════════════════
def render_overview(df: pd.DataFrame) -> None:
    st.title(t("ov_title"))

    rev_share = df.loc[df["Is_Bulk"] == 1, "TotalRevenue"].sum() / df["TotalRevenue"].sum() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("ov_inv"), f"{len(df):,}", t("ov_inv_d"))
    c2.metric(t("ov_ratio"), f"{df['Is_Bulk'].mean() * 100:.1f}%", t("ov_ratio_d"))
    c3.metric(t("ov_aov"), f"£{df['TotalRevenue'].mean():,.2f}", t("ov_aov_d"))
    c4.metric(t("ov_rev"), f"{rev_share:.1f}%", t("ov_rev_d"))

    left, right = st.columns(2)
    with left:
        counts = df["Segment"].value_counts().reset_index()
        counts.columns = ["Segment", "Orders"]
        show(px.pie(counts, names="Segment", values="Orders", hole=0.55, color="Segment",
                    color_discrete_map=SEG_COLORS, title=t("ov_pie_orders")), 340)
    with right:
        rev = df.groupby("Segment")["TotalRevenue"].sum().reset_index()
        show(px.pie(rev, names="Segment", values="TotalRevenue", hole=0.55, color="Segment",
                    color_discrete_map=SEG_COLORS, title=t("ov_pie_rev")), 340)

    card(t("ov_pipe_card"))

    a, b = st.columns(2)
    with a:
        card(t("ov_bulk_card"))
    with b:
        card(t("ov_retail_card"))


def tab_separability(df: pd.DataFrame, sep: pd.DataFrame, features: list[str]) -> None:
    with st.expander(t("sep_how")):
        st.markdown(metric_guide())

    notes = build_takeaways(sep)
    if notes:
        st.markdown(t("sep_takeaways"))
        for n in notes:
            st.markdown(f"- {n}")

    table = sep.assign(Power=sep["Power"].map(POWER_LABEL), Basis=sep["Basis"].map(BASIS))
    st.dataframe(
        table, hide_index=True, use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn(t("col_rank"), width="small"),
            "Feature": st.column_config.TextColumn(t("col_feature")),
            "Power": st.column_config.TextColumn(t("power"), width="small"),
            "Basis": st.column_config.TextColumn(t("col_basis"), help=t("col_basis_help")),
            "Bulk": st.column_config.NumberColumn(t("bulk"), format="%.2f"),
            "Retail": st.column_config.NumberColumn(t("retail"), format="%.2f"),
            "Ratio (B/R)": st.column_config.NumberColumn(t("col_ratio"), format="%.2fx"),
            "KS": st.column_config.ProgressColumn("KS", min_value=0.0, max_value=1.0, format="%.3f"),
            "JS": st.column_config.ProgressColumn("JS", min_value=0.0, max_value=1.0, format="%.3f"),
            "KL": st.column_config.NumberColumn("KL (B‖R)", format="%.2f"),
            "Cohen d": st.column_config.NumberColumn(t("col_cohen"), format="%+.2f"),
            "Mutual Info": st.column_config.NumberColumn(t("col_mi"), format="%.3f"),
            "p-value": st.column_config.NumberColumn(t("col_p"), format="%.1e"),
        },
    )
    st.download_button(t("download_csv"), sep.to_csv(index=False).encode("utf-8-sig"),
                       "feature_separability.csv", "text/csv")

    c1, c2 = st.columns(2)
    with c1:
        show(fig_ks_ranking(sep), 380)
    with c2:
        show(fig_metric_heatmap(sep), 380)
    c3, c4 = st.columns(2)
    with c3:
        show(fig_effect_size(sep), 380)
    with c4:
        show(fig_ratio(sep), 380)

    with st.expander(t("mean_iqr")):
        st.dataframe(segment_comparison_table(df, features), hide_index=True, use_container_width=True)


def tab_relationships(df: pd.DataFrame) -> None:
    method = st.radio(t("rel_method"), ["pearson", "spearman"], horizontal=True,
                      format_func=lambda m: t(m), help=t("rel_method_help"))
    corr = compute_correlations(df, method)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        show(fig_corr_heatmap(corr, method), 460)
    with c2:
        show(fig_target_corr(corr), 280)
        st.markdown(t("top_pairs"))
        st.dataframe(top_feature_pairs(corr), hide_index=True, use_container_width=True,
                     column_config={t("col_corr"): st.column_config.NumberColumn(format="%+.2f")})

    st.markdown(t("explorer"))
    numeric = [c for c in ["Quantity", "UnitPrice", "Unique_SKUs", "TotalRevenue", "Hour"] if c in df.columns]
    e1, e2, e3 = st.columns([1, 1, 1])
    x = e1.selectbox(t("x_axis"), numeric, index=0, format_func=flabel)
    y = e2.selectbox(t("y_axis"), numeric, index=min(2, len(numeric) - 1), format_func=flabel)
    log = e3.checkbox(t("log_scale"), value=True)
    show(fig_relationship(df, x, y, log), 520)

    show(fig_parallel(df, ["Quantity", "Unique_SKUs", "UnitPrice", "Hour", "DayOfWeek", "Month"]), 460)

    with st.expander(t("scatter_matrix")):
        try:
            s = df.dropna(subset=CORE_FEATURES).sample(n=min(1200, len(df)), random_state=42).copy()
            for c in ("Quantity", "UnitPrice"):
                s[c] = np.clip(s[c], s[c].quantile(0.01), s[c].quantile(0.98))
            fig = px.scatter_matrix(s, dimensions=CORE_FEATURES, color="Segment", opacity=0.65,
                                    color_discrete_map=SEG_COLORS, labels={"Segment": t("segment")},
                                    title=t("scatter_matrix_title"))
            fig.update_traces(diagonal_visible=False, marker=dict(size=4))
            show(fig, 680)
        except Exception as exc:
            st.warning(t("scatter_err", err=exc))


def tab_distributions(df: pd.DataFrame, sep_all: pd.DataFrame, importances: dict) -> None:
    c1, c2 = st.columns([1, 1])
    feat = c1.selectbox(t("dist_feature"), FEATURE_COLS, index=0, format_func=flabel)
    kind = c2.radio(t("dist_view"), ["Violin", "KDE", "Histogram"], horizontal=True,
                    format_func=lambda k: {"Violin": t("view_violin"), "KDE": t("view_kde"),
                                           "Histogram": t("view_hist")}[k])
    st.caption(t(f"fd_{feat}"))

    row = sep_all[sep_all["Feature"] == feat]
    if not row.empty:
        r = row.iloc[0]
        basis = BASIS[r["Basis"]]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t("ks_stat"), f"{r['KS']:.3f}", POWER_LABEL[r["Power"]], delta_color="off")
        m2.metric(f"{t('bulk')} ({basis})", f"{r['Bulk']:,.2f}")
        m3.metric(f"{t('retail')} ({basis})", f"{r['Retail']:,.2f}")
        m4.metric(t("rf_imp"), f"{importances.get(feat, 0) * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        show(fig_distribution(df, feat, kind), 420)
    with right:
        show(fig_ecdf(df, feat), 420)
    st.caption(t("dist_cap"))


def tab_time_market(df: pd.DataFrame) -> None:
    show(fig_bulk_heatmap(df), 380)

    c1, c2 = st.columns(2)
    with c1:
        tbl = share_table(df, "DayOfWeek", DAY_NAMES)
        show(px.line(tbl, x="Label", y=SHARE, color="Segment", markers=True, category_orders=SEG_ORDER,
                     color_discrete_map=SEG_COLORS, labels={"Label": t("day")},
                     title=t("weekly_title")), 340)
    with c2:
        tbl = share_table(df, "Month", MONTH_NAMES)
        show(px.bar(tbl, x="Label", y=SHARE, color="Segment", barmode="group", category_orders=SEG_ORDER,
                    color_discrete_map=SEG_COLORS, labels={"Label": t("dist_feature")},
                    title=t("monthly_title")), 340)

    c3, c4 = st.columns(2)
    with c3:
        flags = (df.groupby("Segment")[["Is_UK", "Is_Guest"]].mean() * 100).reset_index()
        flags = flags.melt(id_vars="Segment", var_name="Flag", value_name=SHARE)
        flags["Flag"] = flags["Flag"].map({"Is_UK": t("flag_uk"), "Is_Guest": t("flag_guest")})
        show(px.bar(flags, x="Flag", y=SHARE, color="Segment", barmode="group",
                    color_discrete_map=SEG_COLORS, labels={"Flag": t("flag")},
                    title=t("flags_title")), 340)
    with c4:
        feat = st.selectbox(t("hourly_sel"), ["Quantity", "Unique_SKUs", "UnitPrice", "TotalRevenue"],
                            format_func=flabel)
        hr = df.groupby(["Segment", "Hour"])[feat].mean().reset_index().sort_values("Hour")
        show(px.line(hr, x="Hour", y=feat, color="Segment", markers=True,
                     color_discrete_map=SEG_COLORS, labels={"Hour": t("hour_of_day")},
                     title=t("hourly_title", feat=feat)), 300)


def tab_model(df: pd.DataFrame, model, sep_all: pd.DataFrame, importances: dict) -> None:
    imp_col = t("imp_col")
    imp = pd.DataFrame({"Feature": FEATURE_COLS,
                        imp_col: [round(importances.get(f, 0) * 100, 2) for f in FEATURE_COLS]})
    c1, c2 = st.columns(2)
    with c1:
        show(px.bar(imp.sort_values(imp_col), x=imp_col, y="Feature", orientation="h",
                    color=imp_col, color_continuous_scale="Blues",
                    title=t("imp_title")).update_layout(coloraxis_showscale=False), 400)
    with c2:
        m = sep_all.merge(imp, on="Feature")
        m["Power"] = m["Power"].map(PW)
        fig = px.scatter(m, x="KS", y=imp_col, text="Feature", color="Power",
                         color_discrete_map=POWER_COLORS, title=t("stat_vs_model"))
        fig.update_traces(textposition="top center", marker=dict(size=14))
        fig.update_xaxes(range=[0, 1.1])
        show(fig, 400)
    st.caption(t("model_cap"))

    with st.spinner(t("pca_spin")):
        fig = fig_pca_boundary(df, model)
    if fig is None:
        st.warning(t("pca_warn"))
    else:
        show(fig, 600)
        st.caption(t("pca_cap"))


def render_analytics(df: pd.DataFrame, model) -> None:
    st.title(t("an_title"))

    scope = st.radio(t("an_scope"), ["core", "all"], horizontal=True,
                     format_func=lambda s: t("an_scope_core") if s == "core" else t("an_scope_all"))
    features = CORE_FEATURES if scope == "core" else FEATURE_COLS

    sep = compute_separability(df, tuple(features))
    sep_all = compute_separability(df, tuple(FEATURE_COLS))
    importances = get_feature_importances(model, FEATURE_COLS)

    t1, t2, t3, t4, t5 = st.tabs([t("tab_sep"), t("tab_rel"), t("tab_dist"), t("tab_time"), t("tab_model")])
    with t1:
        tab_separability(df, sep, features)
    with t2:
        tab_relationships(df)
    with t3:
        tab_distributions(df, sep_all, importances)
    with t4:
        tab_time_market(df)
    with t5:
        tab_model(df, model, sep_all, importances)


def render_classifier(model) -> None:
    st.title(t("cl_title"))
    card(t("cl_intro"))

    with st.form("order_input_form"):
        left, right = st.columns(2)
        with left:
            quantity = st.number_input(t("cl_qty"), min_value=1, value=75, step=1)
            unit_price = st.number_input(t("cl_price"), min_value=0.01, value=12.50, format="%.2f")
            unique_skus = st.number_input(t("cl_skus"), min_value=1, value=18, step=1)
            is_uk = st.selectbox(t("cl_market"), [1, 0],
                                 format_func=lambda x: t("cl_uk") if x == 1 else t("cl_intl"))
        with right:
            is_guest = st.selectbox(t("cl_checkout"), [0, 1],
                                    format_func=lambda x: t("cl_reg") if x == 0 else t("cl_guest"))
            hour = st.slider(t("cl_hour"), 0, 23, 14)
            day = st.selectbox(t("cl_day"), list(range(7)), index=2, format_func=lambda x: DAY_FULL[x])
            month = st.slider(t("cl_month"), 1, 12, 11)
        submitted = st.form_submit_button(t("cl_submit"), use_container_width=True)

    if not submitted:
        return

    X = pd.DataFrame([{
        "Quantity": quantity, "UnitPrice": unit_price, "Is_UK": is_uk, "Is_Guest": is_guest,
        "Hour": hour, "DayOfWeek": day, "Month": month, "Unique_SKUs": unique_skus,
    }])[FEATURE_COLS]
    probs = model.predict_proba(X)[0]
    is_bulk = int(model.predict(X)[0]) == 1
    confidence = (probs[1] if is_bulk else probs[0]) * 100
    color = SEG_COLORS[BULK] if is_bulk else SEG_COLORS[RETAIL]
    title = t("cl_bulk_title") if is_bulk else t("cl_retail_title")
    revenue = quantity * unit_price

    c1, c2 = st.columns(2)
    with c1:
        card(f"""
            <div style="text-align:center; padding: 24px 0;">
              <h4>{t("cl_verdict")}</h4>
              <h1 style="color:{color}; font-size:2.2rem; margin:15px 0;">{title}</h1>
              <p style="font-size:1.1rem;">{t("cl_value")}: <b>£{revenue:,.2f}</b><br>
              {t("cl_p_bulk")}: <b>{probs[1] * 100:.2f}%</b> &nbsp;|&nbsp;
              {t("cl_p_retail")}: <b>{probs[0] * 100:.2f}%</b></p>
            </div>""", border=color)
        rule_bulk = quantity >= 50 or revenue >= 500
        agree = rule_bulk == is_bulk
        st.caption(t("cl_rule", rule=BULK if rule_bulk else RETAIL,
                     verdict=t("cl_consistent") if agree else t("cl_differs")))
    with c2:
        show(fig_gauge(confidence, color), 320)


def main() -> None:
    inject_css()

    df, data_key = load_ecommerce_data()
    model, model_key = load_model(df)
    # The cached dataframe is language independent; the display label is added here.
    df = df.assign(Segment=df["Is_Bulk"].map({1: BULK, 0: RETAIL}))

    data_source = t("src_real") if data_key == "real" else t("src_synth")
    model_source = t("model_loaded", path=MODEL_PATH) if model_key == "loaded" else t("model_trained")

    # --- sidebar -----------------------------------------------------------
    st.sidebar.radio(
        t("language"), ["en", "ar"], key="lang", horizontal=True,
        format_func=lambda c: "English" if c == "en" else "العربية",
    )
    st.sidebar.title(t("nav_title"))
    st.sidebar.caption(t("nav_caption"))
    page = st.sidebar.radio(
        t("nav_label"), ["overview", "analytics", "classifier"], key="page",
        format_func=lambda p: t(f"nav_{p}"),
    )
    st.sidebar.divider()
    st.sidebar.caption(t("side_data", src=data_source, n=len(df)))
    st.sidebar.caption(t("side_model", src=model_source))

    if page == "overview":
        render_overview(df)
    elif page == "analytics":
        render_analytics(df, model)
    else:
        render_classifier(model)


main()
