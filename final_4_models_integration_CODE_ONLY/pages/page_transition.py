from __future__ import annotations
import streamlit as st
import streamlit.components.v1 as components


def inject_page_transition() -> None:
    st.markdown("""
    <style>
    :root{--ishara-ms:850ms;--ishara-ease:cubic-bezier(.16,1,.3,1);}
    [data-testid="stAppViewContainer"] .main .block-container{
      animation:isharaVaporIn var(--ishara-ms) var(--ishara-ease) both;
      transform-origin:center top;will-change:opacity,transform,filter;
    }
    [data-testid="stSidebar"]{animation:isharaSidebarIn 720ms var(--ishara-ease) both;}
    header[data-testid="stHeader"]{animation:isharaHeaderIn 650ms var(--ishara-ease) both;}
    .stApp::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:999998;opacity:0;
      background:radial-gradient(circle at 20% 20%,rgba(255,42,42,.22),transparent 32%),radial-gradient(circle at 80% 18%,rgba(255,255,255,.13),transparent 25%),linear-gradient(135deg,rgba(3,7,18,.76),rgba(3,7,18,.05));
      backdrop-filter:blur(18px) saturate(1.05);-webkit-backdrop-filter:blur(18px) saturate(1.05);
      animation:isharaMist 950ms var(--ishara-ease) both;}
    body.ishara-page-leaving [data-testid="stAppViewContainer"] .main .block-container{
      animation:isharaVaporOut 520ms cubic-bezier(.7,0,.84,0) both!important;}
    body.ishara-page-leaving .stApp::after{content:"";position:fixed;inset:0;z-index:999999;pointer-events:none;
      background:radial-gradient(circle at 50% 45%,rgba(255,42,42,.20),transparent 35%),linear-gradient(135deg,rgba(3,7,18,.08),rgba(3,7,18,.78));
      backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);animation:isharaLeaveMist 560ms cubic-bezier(.7,0,.84,0) both;}
    [data-testid="stVerticalBlock"]>div,.stTabs,.stDataFrame,.stMarkdown,.stButton,.stTextInput,.stTextArea,.stSelectbox,.stFileUploader,.stExpander,.stAlert,.stMetric,.stImage,.stAudio,.stVideo{
      animation:isharaElementIn 760ms var(--ishara-ease) both;will-change:opacity,transform,filter;}
    [data-testid="stVerticalBlock"]>div:nth-child(1){animation-delay:25ms}[data-testid="stVerticalBlock"]>div:nth-child(2){animation-delay:55ms}[data-testid="stVerticalBlock"]>div:nth-child(3){animation-delay:85ms}[data-testid="stVerticalBlock"]>div:nth-child(4){animation-delay:115ms}[data-testid="stVerticalBlock"]>div:nth-child(5){animation-delay:145ms}[data-testid="stVerticalBlock"]>div:nth-child(6){animation-delay:175ms}[data-testid="stVerticalBlock"]>div:nth-child(n+7){animation-delay:205ms}
    @keyframes isharaVaporIn{0%{opacity:0;transform:translateY(24px) scale(.982);filter:blur(24px) saturate(.65)}45%{opacity:.72;transform:translateY(8px) scale(.994);filter:blur(9px) saturate(.9)}100%{opacity:1;transform:translateY(0) scale(1);filter:blur(0) saturate(1)}}
    @keyframes isharaElementIn{0%{opacity:0;transform:translateY(14px);filter:blur(10px)}100%{opacity:1;transform:translateY(0);filter:blur(0)}}
    @keyframes isharaVaporOut{0%{opacity:1;transform:translateY(0) scale(1);filter:blur(0)}100%{opacity:0;transform:translateY(-18px) scale(.985);filter:blur(22px) saturate(.65)}}
    @keyframes isharaMist{0%{opacity:1;transform:scale(1.03)}45%{opacity:.45}100%{opacity:0;transform:scale(1)}}
    @keyframes isharaLeaveMist{0%{opacity:0;transform:scale(1)}45%{opacity:.85;transform:scale(1.01)}100%{opacity:1;transform:scale(1.02)}}
    @keyframes isharaSidebarIn{0%{opacity:0;transform:translateX(-14px);filter:blur(12px)}100%{opacity:1;transform:translateX(0);filter:blur(0)}}
    @keyframes isharaHeaderIn{0%{opacity:0;transform:translateY(-12px);filter:blur(10px)}100%{opacity:1;transform:translateY(0);filter:blur(0)}}
    @media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:1ms!important;animation-delay:0ms!important;transition-duration:1ms!important}}
    </style>
    """, unsafe_allow_html=True)
    components.html("""
    <script>
    (function(){try{const d=window.parent.document;if(d.getElementById('ishara-transition-js'))return;const m=d.createElement('script');m.id='ishara-transition-js';m.type='text/plain';m.textContent='installed';d.head.appendChild(m);d.addEventListener('click',function(e){const a=e.target&&e.target.closest?e.target.closest('a'):null;if(!a)return;const href=a.getAttribute('href')||'';if(!href||href.startsWith('#')||a.target==='_blank'||e.ctrlKey||e.metaKey||e.shiftKey||e.altKey)return;d.body.classList.add('ishara-page-leaving');setTimeout(()=>d.body.classList.remove('ishara-page-leaving'),900);},true);}catch(e){}})();
    </script>
    """, height=0)
