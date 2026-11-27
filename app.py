"""
Decision Tree Studio  (English / Arabic)
-----------------------------------------
صفحة "استوديو شجرة القرار" — تتركّب كصفحة رابعة جوه تطبيق E-Commerce Intelligence.

الاستخدام من app.py:
    import dt_studio
    dt_studio.render_decision_tree(df[FEATURE_COLS + ["Is_Bulk"]], FEATURE_COLS, "Is_Bulk", IS_AR)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

TR: dict[str, tuple[str, str]] = {
    "title": ("🌳 Decision Tree Studio", "🌳 استوديو شجرة القرار"),
    "intro": ("<p>Train an interpretable Decision Tree on the order data (or your own CSV), tune its "
              "hyperparameters, and inspect the tree, errors and feature importance.</p>",
              "<p>درّب شجرة قرار قابلة للتفسير على بيانات الطلبات (أو ملف CSV بتاعك)، واضبط إعداداتها، "
              "وافحص الشجرة والأخطاء وأهمية الخصائص.</p>"),
    "source": ("Data source", "مصدر البيانات"),
    "src_app": ("App data (Is_Bulk target)", "بيانات التطبيق (الهدف Is_Bulk)"),
    "src_upload": ("Upload CSV", "رفع ملف CSV"),
    "upload_lbl": ("Upload dataset (last column must be the target)", "ارفع البيانات (آخر عمود لازم يكون الهدف)"),
    "upload_hint": ("No file uploaded yet — using the app data.", "لسه مفيش ملف مرفوع — هستخدم بيانات التطبيق."),
    "no_numeric": ("The uploaded file needs at least one numeric feature column and a target column.",
                   "الملف المرفوع لازم فيه عمود خاصية رقمي واحد على الأقل وعمود هدف."),
    "hp": ("⚙️ Model settings", "⚙️ إعدادات النموذج"),
    "criterion": ("Split criterion", "معيار التقسيم"),
    "depth": ("Max depth", "أقصى عمق"),
    "min_split": ("Min samples split", "أقل عدد عينات للتقسيم"),
    "test_size": ("Test size", "نسبة الاختبار"),
    "m_test": ("Test accuracy", "دقة الاختبار"),
    "m_train": ("Train accuracy", "دقة التدريب"),
    "m_samples": ("Samples", "عدد العينات"),
    "m_depth": ("Actual depth", "العمق الفعلي"),
    "overfit": ("⚠️ Train accuracy is {gap:.1f} points above test accuracy — the tree is likely overfitting. "
                "Try a smaller max depth or a larger min samples split.",
                "⚠️ دقة التدريب أعلى من دقة الاختبار بـ {gap:.1f} نقطة — غالبًا الشجرة بتعمل overfitting. "
                "جرّب عمق أقل أو قيمة أكبر لأقل عدد عينات للتقسيم."),
    "leak": ("ℹ️ `Quantity` defines the Bulk label (≥ 50 units or ≥ £500), so very high accuracy here is "
             "expected and is an upper bound. Remove `Quantity` from the features to test real generalisation.",
             "ℹ️ `Quantity` هي اللي بتعرّف تصنيف الجملة (≥ 50 وحدة أو ≥ £500)، فالدقة العالية جدًا متوقعة وهي "
             "حد أعلى. شيل `Quantity` من الخصائص علشان تختبر التعميم الحقيقي."),
    "feat_pick": ("Features used", "الخصائص المستخدمة"),
    "tab_pred": ("🎯 Prediction", "🎯 التنبؤ"),
    "tab_tree": ("🌲 Tree", "🌲 الشجرة"),
    "tab_eval": ("📏 Evaluation", "📏 التقييم"),
    "tab_data": ("🗂️ Data", "🗂️ البيانات"),
    "pred_head": ("Enter feature values", "أدخل قيم الخصائص"),
    "value_for": ("Value for {c}", "قيمة {c}"),
    "run": ("Run prediction", "تشغيل التنبؤ"),
    "pred_class": ("Predicted class", "الفئة المتوقعة"),
    "pred_conf": ("Confidence", "الثقة"),
    "tree_levels": ("Levels to display", "عدد المستويات المعروضة"),
    "tree_cap": ("Only the top levels are drawn so the plot stays readable; the model itself uses the full depth.",
                 "بنعرض المستويات العليا بس علشان الرسم يفضل مقروء؛ النموذج نفسه بيستخدم العمق الكامل."),
    "cm": ("Confusion matrix", "مصفوفة الالتباس"),
    "cm_x": ("Predicted", "المتوقع"),
    "cm_y": ("Actual", "الفعلي"),
    "fi": ("Feature importance", "أهمية الخصائص"),
    "fi_col": ("Importance (%)", "الأهمية (%)"),
    "cr": ("Classification report", "تقرير التصنيف"),
    "preview": ("Dataset preview", "معاينة البيانات"),
}


def render_decision_tree(df: pd.DataFrame, feature_cols: list[str], target_col: str, is_ar: bool = False) -> None:
    def t(key: str, **kw) -> str:
        s = TR[key][1 if is_ar else 0]
        return s.format(**kw) if kw else s

    st.title(t("title"))
    st.markdown(f'<div class="glass-container">{t("intro")}</div>', unsafe_allow_html=True)

    # ── data source ────────────────────────────────────────────────────────
    data = df[feature_cols + [target_col]].copy()
    uses_app_data = True
    source = st.radio(t("source"), ["app", "upload"], horizontal=True, key="dt_source",
                      format_func=lambda s: t("src_app") if s == "app" else t("src_upload"))
    if source == "upload":
        up = st.file_uploader(t("upload_lbl"), type=["csv"], key="dt_upload")
        if up is None:
            st.info(t("upload_hint"))
        else:
            raw = pd.read_csv(up)
            feats = raw.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            if feats.shape[1] == 0 or raw.shape[1] < 2:
                st.error(t("no_numeric"))
                return
            data = feats.join(raw.iloc[:, -1]).dropna()
            target_col = raw.columns[-1]
            uses_app_data = False

    all_feats = [c for c in data.columns if c != target_col]

    # ── hyper-parameters ───────────────────────────────────────────────────
    with st.expander(t("hp"), expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        criterion = c1.selectbox(t("criterion"), ["gini", "entropy", "log_loss"], key="dt_crit")
        max_depth = c2.slider(t("depth"), 1, 15, 4, key="dt_depth")
        min_split = c3.slider(t("min_split"), 2, 20, 2, key="dt_split")
        test_size = c4.slider(t("test_size"), 0.1, 0.5, 0.2, 0.05, key="dt_test")
        features = st.multiselect(t("feat_pick"), all_feats, default=all_feats, key=f"dt_feats_{uses_app_data}")

    if not features:
        return

    X, y = data[features], data[target_col]
    stratify = y if y.nunique() <= 20 and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                   min_samples_split=min_split, random_state=42).fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    train_acc = accuracy_score(y_tr, model.predict(X_tr)) * 100
    test_acc = accuracy_score(y_te, y_pred) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("m_test"), f"{test_acc:.2f}%")
    m2.metric(t("m_train"), f"{train_acc:.2f}%")
    m3.metric(t("m_samples"), f"{len(data):,}")
    m4.metric(t("m_depth"), f"{model.get_depth()}")

    if train_acc - test_acc > 10:
        st.warning(t("overfit", gap=train_acc - test_acc))
    if uses_app_data and "Quantity" in features:
        st.caption(t("leak"))

    tab1, tab2, tab3, tab4 = st.tabs([t("tab_pred"), t("tab_tree"), t("tab_eval"), t("tab_data")])

    # ── prediction ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader(t("pred_head"))
        values, cols = {}, st.columns(2)
        for i, c in enumerate(features):
            lo, hi, mean = float(X[c].min()), float(X[c].max()), float(X[c].mean())
            span = (hi - lo) or 1.0
            with cols[i % 2]:
                values[c] = st.number_input(t("value_for", c=c), min_value=lo - span, max_value=hi + span,
                                            value=mean, format="%.4f", key=f"dt_in_{c}")
        if st.button(t("run"), key="dt_run"):
            row = pd.DataFrame([values])[features]
            pred = model.predict(row)[0]
            conf = model.predict_proba(row)[0].max() * 100
            st.markdown(
                f"""<div class="glass-container" style="border-left:5px solid #39FF14;">
                <h3 style="margin:0;">{t("pred_class")}: <span style="color:#00C6FF;">{pred}</span></h3>
                <p style="margin:6px 0 0 0;">{t("pred_conf")}: <b>{conf:.2f}%</b></p></div>""",
                unsafe_allow_html=True)

    # ── tree ───────────────────────────────────────────────────────────────
    with tab2:
        levels = st.slider(t("tree_levels"), 1, max(model.get_depth(), 1), min(3, max(model.get_depth(), 1)),
                           key="dt_levels")
        fig, ax = plt.subplots(figsize=(18, 3 + 2.2 * levels), facecolor="white")
        plot_tree(model, feature_names=features, class_names=[str(c) for c in model.classes_],
                  filled=True, rounded=True, fontsize=9, max_depth=levels, ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(t("tree_cap"))

    # ── evaluation ─────────────────────────────────────────────────────────
    with tab3:
        left, right = st.columns(2)
        with left:
            labels = [str(c) for c in model.classes_]
            cm = confusion_matrix(y_te, y_pred, labels=model.classes_)
            fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, aspect="auto",
                               color_continuous_scale=["#0B2A4A", "#00C6FF", "#39FF14"],
                               labels=dict(x=t("cm_x"), y=t("cm_y")), title=t("cm"))
            fig_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, height=380)
            st.plotly_chart(fig_cm, use_container_width=True)
        with right:
            imp = (pd.DataFrame({"Feature": features, t("fi_col"): model.feature_importances_ * 100})
                   .sort_values(t("fi_col")))
            fig_fi = px.bar(imp, x=t("fi_col"), y="Feature", orientation="h", title=t("fi"),
                            color=t("fi_col"), color_continuous_scale="Blues")
            fig_fi.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, height=380)
            st.plotly_chart(fig_fi, use_container_width=True)

        st.subheader(t("cr"))
        report = pd.DataFrame(classification_report(y_te, y_pred, output_dict=True, zero_division=0)).T
        st.dataframe(report.round(3), use_container_width=True)

    # ── data ───────────────────────────────────────────────────────────────
    with tab4:
        st.subheader(t("preview"))
        st.dataframe(data.head(25), use_container_width=True)
      
