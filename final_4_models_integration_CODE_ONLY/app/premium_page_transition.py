from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def inject_premium_page_transitions(
    *,
    effect: str = "vapor",
    intensity: str = "epic",
    duration_ms: int = 1080,
) -> None:
    """
    Premium global page transitions for Ishara AI.

    Inject once from streamlit_app.py and it affects every Streamlit page:
    login/auth, home, profile, live, emergency and model pages.

    It combines:
    - cinematic CSS transitions
    - a real overlay curtain injected by JS
    - navigation click detection
    - rerun pulse for button-triggered reruns
    - staggered element reveal
    """

    effect = (effect or "vapor").lower().strip()
    intensity = (intensity or "epic").lower().strip()

    if intensity == "soft":
        blur = 14
        mist_opacity = 0.50
        scale_from = 0.992
        y_from = 14
        glow = 0.18
    elif intensity == "epic":
        blur = 34
        mist_opacity = 0.88
        scale_from = 0.968
        y_from = 36
        glow = 0.36
    else:
        blur = 24
        mist_opacity = 0.72
        scale_from = 0.982
        y_from = 24
        glow = 0.26

    duration_ms = max(420, min(int(duration_ms), 1900))

    if effect == "slide":
        enter_transform = f"translateX(38px) translateY({y_from}px) scale({scale_from})"
        exit_transform = "translateX(-46px) translateY(-10px) scale(.982)"
    elif effect == "zoom":
        enter_transform = f"translateY({y_from}px) scale(.93)"
        exit_transform = "translateY(-22px) scale(1.038)"
    else:
        enter_transform = f"translateY({y_from}px) scale({scale_from})"
        exit_transform = "translateY(-26px) scale(.978)"

    st.markdown(
        f"""
        <style id="ishara-premium-page-transitions-css">
        :root {{
            --ishara-transition-duration: {duration_ms}ms;
            --ishara-transition-duration-fast: {max(420, int(duration_ms * 0.60))}ms;
            --ishara-ease-out: cubic-bezier(.16, 1, .3, 1);
            --ishara-ease-in: cubic-bezier(.70, 0, .84, 0);
            --ishara-red: 248, 37, 37;
            --ishara-bg: 2, 6, 23;
            --ishara-panel: 8, 13, 24;
            --ishara-blur: {blur}px;
            --ishara-mist-opacity: {mist_opacity};
            --ishara-glow: {glow};
        }}

        html {{ scroll-behavior: smooth; }}

        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] .main,
        [data-testid="stAppViewContainer"] .main .block-container,
        [data-testid="stSidebar"],
        header[data-testid="stHeader"] {{
            backface-visibility: hidden;
            transform-style: preserve-3d;
        }}

        [data-testid="stAppViewContainer"] .main .block-container {{
            position: relative;
            animation:
                isharaPremiumPageIn var(--ishara-transition-duration) var(--ishara-ease-out) both,
                isharaPremiumClipReveal calc(var(--ishara-transition-duration) + 160ms) var(--ishara-ease-out) both;
            transform-origin: center top;
            will-change: opacity, transform, filter, clip-path;
        }}

        [data-testid="stSidebar"] {{
            animation: isharaPremiumSidebarIn 860ms var(--ishara-ease-out) both;
            will-change: opacity, transform, filter;
        }}

        header[data-testid="stHeader"] {{
            animation: isharaPremiumHeaderIn 680ms var(--ishara-ease-out) both;
            will-change: opacity, transform, filter;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            inset: -8%;
            pointer-events: none;
            z-index: 999990;
            opacity: 0;
            background:
                radial-gradient(circle at 16% 18%, rgba(var(--ishara-red), .34), transparent 22%),
                radial-gradient(circle at 86% 10%, rgba(255,255,255,.16), transparent 18%),
                radial-gradient(circle at 48% 86%, rgba(var(--ishara-red), .18), transparent 34%),
                radial-gradient(circle at 58% 42%, rgba(255,255,255,.08), transparent 26%),
                linear-gradient(135deg, rgba(var(--ishara-bg), .88), rgba(var(--ishara-bg), .04) 62%, rgba(var(--ishara-bg), .68));
            filter: saturate(1.08);
            backdrop-filter: blur(calc(var(--ishara-blur) * .78)) saturate(1.05);
            -webkit-backdrop-filter: blur(calc(var(--ishara-blur) * .78)) saturate(1.05);
            animation: isharaPremiumMistIn calc(var(--ishara-transition-duration) + 120ms) var(--ishara-ease-out) both;
        }}

        .stApp::after {{
            content: "";
            position: fixed;
            inset: -40%;
            z-index: 999989;
            pointer-events: none;
            opacity: 0;
            background:
                linear-gradient(115deg,
                    transparent 0%,
                    transparent 34%,
                    rgba(255,255,255,.08) 42%,
                    rgba(var(--ishara-red), .12) 48%,
                    transparent 58%,
                    transparent 100%);
            transform: translateX(-18%);
            animation: isharaPremiumLightSweep calc(var(--ishara-transition-duration) + 260ms) var(--ishara-ease-out) both;
        }}

        body.ishara-page-leaving [data-testid="stAppViewContainer"] .main .block-container {{
            animation: isharaPremiumPageOut var(--ishara-transition-duration-fast) var(--ishara-ease-in) both !important;
        }}

        body.ishara-page-leaving [data-testid="stSidebar"],
        body.ishara-page-leaving header[data-testid="stHeader"] {{
            animation: isharaPremiumChromeOut var(--ishara-transition-duration-fast) var(--ishara-ease-in) both !important;
        }}

        body.ishara-page-leaving .ishara-transition-curtain {{
            opacity: 1;
            visibility: visible;
            transform: scale(1);
        }}

        body.ishara-page-leaving .ishara-transition-curtain__mist {{
            animation: isharaPremiumLeaveMist var(--ishara-transition-duration-fast) var(--ishara-ease-in) both;
        }}

        body.ishara-page-leaving .ishara-transition-curtain__orb {{
            animation: isharaPremiumLeaveOrb var(--ishara-transition-duration-fast) var(--ishara-ease-in) both;
        }}

        body.ishara-rerun-pending .ishara-transition-curtain {{
            opacity: 1;
            visibility: visible;
        }}

        body.ishara-rerun-pending .ishara-transition-curtain__mist {{
            animation: isharaPremiumPulseMist 780ms var(--ishara-ease-out) infinite alternate;
        }}

        [data-testid="stVerticalBlock"] > div,
        [data-testid="stHorizontalBlock"] > div,
        [data-testid="column"],
        .stMarkdown,
        .stButton,
        .stTextInput,
        .stTextArea,
        .stSelectbox,
        .stMultiSelect,
        .stFileUploader,
        .stExpander,
        .stAlert,
        .stMetric,
        .stImage,
        .stAudio,
        .stVideo,
        .stDataFrame,
        .stTabs,
        .stRadio,
        .stSlider,
        .stCheckbox {{
            animation: isharaPremiumElementIn 780ms var(--ishara-ease-out) both;
            will-change: opacity, transform, filter;
        }}

        [data-testid="stVerticalBlock"] > div:nth-child(1) {{ animation-delay: 35ms; }}
        [data-testid="stVerticalBlock"] > div:nth-child(2) {{ animation-delay: 70ms; }}
        [data-testid="stVerticalBlock"] > div:nth-child(3) {{ animation-delay: 105ms; }}
        [data-testid="stVerticalBlock"] > div:nth-child(4) {{ animation-delay: 140ms; }}
        [data-testid="stVerticalBlock"] > div:nth-child(5) {{ animation-delay: 175ms; }}
        [data-testid="stVerticalBlock"] > div:nth-child(6) {{ animation-delay: 210ms; }}
        [data-testid="stVerticalBlock"] > div:nth-child(7) {{ animation-delay: 245ms; }}
        [data-testid="stVerticalBlock"] > div:nth-child(8) {{ animation-delay: 280ms; }}
        [data-testid="stVerticalBlock"] > div:nth-child(n+9) {{ animation-delay: 310ms; }}

        .stButton button,
        button[kind],
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"] {{
            transition:
                transform 260ms var(--ishara-ease-out),
                filter 260ms var(--ishara-ease-out),
                box-shadow 260ms var(--ishara-ease-out),
                opacity 260ms var(--ishara-ease-out) !important;
        }}

        .stButton button:hover,
        button[kind]:hover,
        [data-testid="baseButton-secondary"]:hover,
        [data-testid="baseButton-primary"]:hover {{
            transform: translateY(-2px) scale(1.012);
            filter: brightness(1.07);
            box-shadow: 0 14px 36px rgba(var(--ishara-red), calc(var(--ishara-glow) * .55));
        }}

        .ishara-transition-curtain {{
            position: fixed;
            inset: 0;
            z-index: 2147483000;
            pointer-events: none;
            opacity: 0;
            visibility: hidden;
            transition:
                opacity 220ms var(--ishara-ease-out),
                visibility 220ms var(--ishara-ease-out);
            overflow: hidden;
        }}

        .ishara-transition-curtain__mist {{
            position: absolute;
            inset: -10%;
            background:
                radial-gradient(circle at 22% 28%, rgba(var(--ishara-red), .32), transparent 26%),
                radial-gradient(circle at 74% 18%, rgba(255,255,255,.16), transparent 18%),
                radial-gradient(circle at 50% 64%, rgba(var(--ishara-red), .16), transparent 32%),
                linear-gradient(135deg, rgba(var(--ishara-bg), .10), rgba(var(--ishara-bg), .88));
            backdrop-filter: blur(calc(var(--ishara-blur) * .72));
            -webkit-backdrop-filter: blur(calc(var(--ishara-blur) * .72));
        }}

        .ishara-transition-curtain__orb {{
            position: absolute;
            width: min(52vw, 620px);
            aspect-ratio: 1 / 1;
            border-radius: 999px;
            left: 50%;
            top: 48%;
            transform: translate(-50%, -50%) scale(.62);
            background:
                radial-gradient(circle, rgba(255,255,255,.38) 0%, rgba(var(--ishara-red),.23) 28%, rgba(var(--ishara-bg),.02) 62%, transparent 72%);
            filter: blur(18px);
            opacity: 0;
        }}

        .ishara-transition-curtain__grain {{
            position: absolute;
            inset: 0;
            opacity: .16;
            mix-blend-mode: overlay;
            background-image:
                repeating-radial-gradient(circle at 20% 30%, rgba(255,255,255,.10) 0 1px, transparent 1px 5px),
                repeating-linear-gradient(90deg, rgba(255,255,255,.045) 0 1px, transparent 1px 8px);
            animation: isharaPremiumGrain 780ms steps(2, end) infinite;
        }}

        @keyframes isharaPremiumPageIn {{
            0% {{
                opacity: 0;
                transform: {enter_transform};
                filter: blur(var(--ishara-blur)) saturate(.62) contrast(.92);
            }}
            42% {{
                opacity: .76;
                filter: blur(calc(var(--ishara-blur) * .38)) saturate(.86) contrast(.96);
            }}
            100% {{
                opacity: 1;
                transform: translateX(0) translateY(0) scale(1);
                filter: blur(0) saturate(1) contrast(1);
            }}
        }}

        @keyframes isharaPremiumClipReveal {{
            0% {{ clip-path: inset(6% 2% 0 2% round 28px); }}
            100% {{ clip-path: inset(0 0 0 0 round 0); }}
        }}

        @keyframes isharaPremiumPageOut {{
            0% {{
                opacity: 1;
                transform: translateX(0) translateY(0) scale(1);
                filter: blur(0) saturate(1);
            }}
            100% {{
                opacity: 0;
                transform: {exit_transform};
                filter: blur(calc(var(--ishara-blur) * .95)) saturate(.55);
            }}
        }}

        @keyframes isharaPremiumElementIn {{
            0% {{
                opacity: 0;
                transform: translateY(18px) scale(.992);
                filter: blur(12px);
            }}
            100% {{
                opacity: 1;
                transform: translateY(0) scale(1);
                filter: blur(0);
            }}
        }}

        @keyframes isharaPremiumMistIn {{
            0% {{ opacity: var(--ishara-mist-opacity); transform: scale(1.04) translateY(0); }}
            45% {{ opacity: calc(var(--ishara-mist-opacity) * .42); }}
            100% {{ opacity: 0; transform: scale(1) translateY(-1%); }}
        }}

        @keyframes isharaPremiumLightSweep {{
            0% {{ opacity: 0; transform: translateX(-26%); }}
            18% {{ opacity: .70; }}
            100% {{ opacity: 0; transform: translateX(28%); }}
        }}

        @keyframes isharaPremiumLeaveMist {{
            0% {{ opacity: 0; transform: scale(1.00) translateY(2%); filter: blur(0); }}
            100% {{ opacity: 1; transform: scale(1.035) translateY(0); filter: blur(2px); }}
        }}

        @keyframes isharaPremiumLeaveOrb {{
            0% {{ opacity: 0; transform: translate(-50%, -50%) scale(.58); }}
            72% {{ opacity: .82; }}
            100% {{ opacity: .55; transform: translate(-50%, -50%) scale(1.12); }}
        }}

        @keyframes isharaPremiumPulseMist {{
            0% {{ opacity: .34; transform: scale(1); }}
            100% {{ opacity: .68; transform: scale(1.02); }}
        }}

        @keyframes isharaPremiumSidebarIn {{
            0% {{ opacity: 0; transform: translateX(-22px); filter: blur(16px); }}
            100% {{ opacity: 1; transform: translateX(0); filter: blur(0); }}
        }}

        @keyframes isharaPremiumHeaderIn {{
            0% {{ opacity: 0; transform: translateY(-14px); filter: blur(12px); }}
            100% {{ opacity: 1; transform: translateY(0); filter: blur(0); }}
        }}

        @keyframes isharaPremiumChromeOut {{
            0% {{ opacity: 1; filter: blur(0); }}
            100% {{ opacity: .18; filter: blur(10px); }}
        }}

        @keyframes isharaPremiumGrain {{
            0% {{ transform: translate(0, 0); }}
            25% {{ transform: translate(-1%, 1%); }}
            50% {{ transform: translate(1%, -1%); }}
            75% {{ transform: translate(-.5%, -.5%); }}
            100% {{ transform: translate(0, 0); }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                animation-duration: 1ms !important;
                animation-delay: 0ms !important;
                transition-duration: 1ms !important;
            }}
            .ishara-transition-curtain {{ display: none !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    components.html(
        """
        <script>
        (function () {
            const TRANSITION_MS = 680;
            const COOLDOWN_MS = 960;

            function getParentDoc() {
                try { return window.parent.document; }
                catch (e) { return document; }
            }

            const doc = getParentDoc();

            function ensureCurtain() {
                let curtain = doc.querySelector(".ishara-transition-curtain");
                if (curtain) return curtain;

                curtain = doc.createElement("div");
                curtain.className = "ishara-transition-curtain";
                curtain.innerHTML = `
                    <div class="ishara-transition-curtain__mist"></div>
                    <div class="ishara-transition-curtain__orb"></div>
                    <div class="ishara-transition-curtain__grain"></div>
                `;
                doc.body.appendChild(curtain);
                return curtain;
            }

            function startLeave() {
                ensureCurtain();
                if (doc.body.dataset.isharaTransitionLock === "1") return;
                doc.body.dataset.isharaTransitionLock = "1";
                doc.body.classList.add("ishara-page-leaving");

                setTimeout(function () {
                    doc.body.classList.remove("ishara-page-leaving");
                }, TRANSITION_MS + 220);

                setTimeout(function () {
                    doc.body.dataset.isharaTransitionLock = "0";
                }, COOLDOWN_MS);
            }

            function startRerunPulse() {
                ensureCurtain();
                doc.body.classList.add("ishara-rerun-pending");
                setTimeout(function () {
                    doc.body.classList.remove("ishara-rerun-pending");
                }, 1300);
            }

            function looksLikeNavTarget(el) {
                if (!el) return false;

                const anchor = el.closest && el.closest("a");
                if (anchor) {
                    const href = anchor.getAttribute("href") || "";
                    if (href && !href.startsWith("#") && !href.startsWith("mailto:") && !href.startsWith("tel:")) {
                        return true;
                    }
                }

                const sidebar = el.closest && el.closest('[data-testid="stSidebar"]');
                if (!sidebar) return false;

                const text = (el.innerText || el.textContent || "").trim();
                if (!text) return false;

                if (el.closest('[role="button"]')) return true;
                if (el.closest("button")) return true;
                if (el.closest('[data-testid*="stPageLink"]')) return true;
                if (el.closest('[data-testid*="stSidebarNav"]')) return true;
                if (el.closest('[data-testid*="stNav"]')) return true;
                return false;
            }

            if (!doc.body.dataset.isharaPremiumTransitionsInstalled) {
                doc.body.dataset.isharaPremiumTransitionsInstalled = "1";
                ensureCurtain();

                doc.addEventListener("click", function (event) {
                    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
                    if (looksLikeNavTarget(event.target)) {
                        startLeave();
                    } else {
                        const btn = event.target && event.target.closest ? event.target.closest("button") : null;
                        if (btn && !btn.closest('[data-testid="stSidebar"]')) // startRerunPulse disabled for lighter local demo;
                    }
                }, true);

                doc.addEventListener("keydown", function (event) {
                    if (event.key === "Enter" && looksLikeNavTarget(doc.activeElement)) startLeave();
                }, true);

                const observer = new MutationObserver(function () {
                    if (doc.body.classList.contains("ishara-rerun-pending")) {
                        setTimeout(function () { doc.body.classList.remove("ishara-rerun-pending"); }, 380);
                    }
                });
                observer.observe(doc.body, { childList: true, subtree: true });
            }
        })();
        </script>
        """,
        height=0,
    )
