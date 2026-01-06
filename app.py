import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go

# 모듈 임포트
import genesis
import psy_sim_config
import engine
import create_csv_data
import inference
import os

# ==========================================
# Streamlit Configuration
# ==========================================
st.set_page_config(
    page_title="Psy-Sim v2.2 Controller",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Helper Functions
# ==========================================
@st.cache_data
def load_configs():
    # 데이터 생성 확인
    if not os.path.exists('data/activities.csv'):
        create_csv_data.create_initial_csvs()
        
    # v2.2 호환 로더 사용
    df_act = psy_sim_config.load_activity_table()
    
    # Life Pattern 데이터 로드
    df_patterns, _, _ = psy_sim_config.load_life_patterns()
    
    return df_act, df_patterns

# ==========================================
# Sidebar
# ==========================================
st.sidebar.title("🧠 Psy-Sim v2.2")
st.sidebar.caption("Dynamic Economy & Social Simulator")
st.sidebar.markdown("---")

# 1. Population Control
st.sidebar.subheader("👥 Population")
n_agents = st.sidebar.slider("Agent Count", 100, 20000, 5000, 100)

# 2. Event Injection Control (Dynamic World)
st.sidebar.subheader("⚡ World Events")
enable_maintenance = st.sidebar.checkbox("Trigger Server Maintenance (14:00)", value=False)
enable_hottime = st.sidebar.checkbox("Trigger Hot Time (20:00)", value=True)

# 3. Execution
st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Simulation", type="primary")

# ==========================================
# Main Layout
# ==========================================
st.title("📊 Psychological Market Simulator Dashboard")

df_activities, df_patterns = load_configs()

# Data Preview Tabs
tab1, tab2 = st.tabs(["📂 Activity Data", "🧬 Life Patterns"])
with tab1:
    st.dataframe(df_activities, use_container_width=True)
with tab2:
    st.dataframe(df_patterns.head(10), use_container_width=True)
    st.caption("*Showing first 10 rows of 96 ticks per pattern")

# ==========================================
# Simulation Execution
# ==========================================
if run_btn:
    # 1. Setup Events
    events = {}
    if enable_maintenance:
        events[56] = {"Type": "SERVER_DOWN", "Target": "GAME", "Value": 0}
        events[57] = {"Type": "SERVER_DOWN", "Target": "GAME", "Value": 0}
    
    if enable_hottime:
        events[80] = {"Type": "HOT_TIME", "Target": "GAME", "Value": 3.0}

    # 2. Generate Agents
    with st.spinner(f"Creating {n_agents:,} Agents with Life Patterns..."):
        population = genesis.create_agent_population(n_agents)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Agents", f"{n_agents:,}")
    col2.metric("Active Patterns", "4 Types")
    col3.metric("Scheduled Events", len(events))
    
    # 3. Run Engine
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.write("⚙️ Running Physics Engine (24h Loop)...")
    
    start_t = time.time()
    logs = engine.run_simulation(population, df_activities, events=events)
    end_t = time.time()
    
    progress_bar.progress(100)
    status_text.success(f"Simulation finished in {end_t - start_t:.2f}s")
    
    # ==========================================
    # Visualizations
    # ==========================================
    st.markdown("---")
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"{logs['total_revenue'][-1]:,.0f} G")
    m2.metric("Avg Stress", f"{logs['avg_stress'][-1]:.1f}")
    m3.metric("Avg Dopamine", f"{logs['avg_dopamine'][-1]:.1f}")
    m4.metric("Avg Anxiety", f"{logs['avg_anxiety'][-1]:.1f}")

    # --- Chart 1: Main Trends ---
    st.subheader("📈 Macro Trends (24 Hours)")
    df_logs = pd.DataFrame({
        "Time": logs['time'],
        "Revenue": logs['total_revenue'],
        "Stress": logs['avg_stress'],
        "Dopamine": logs['avg_dopamine']
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_logs['Time'], y=df_logs['Stress'], name="Stress", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df_logs['Time'], y=df_logs['Dopamine'], name="Dopamine", line=dict(color='green')))
    fig.update_layout(title="Stress vs Dopamine Levels", xaxis_title="Time", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Chart 2: Life Pattern Stress Comparison ---
    st.subheader("👥 Stress by Life Pattern")
    pattern_names = {0:"Office", 1:"Student", 2:"Free", 3:"Night"}
    df_pattern_stress = pd.DataFrame(logs['pattern_stress'])
    df_pattern_stress['Time'] = logs['time']
    
    fig_p = go.Figure()
    for pid, name in pattern_names.items():
        if pid in df_pattern_stress.columns:
            fig_p.add_trace(go.Scatter(x=df_pattern_stress['Time'], y=df_pattern_stress[pid], name=name))
    fig_p.update_layout(title="Stress Levels per Pattern", xaxis_title="Time")
    st.plotly_chart(fig_p, use_container_width=True)

    # --- Chart 3: Social Viral Trends ---
    st.subheader("🔥 Social Viral Trends (Bandwagon Effect)")
    viral_data = np.array(logs['viral_trends'])
    df_viral = pd.DataFrame(viral_data, columns=inference.MEDIA_TYPES)
    df_viral['Time'] = logs['time']
    
    fig_v = px.line(df_viral, x='Time', y=inference.MEDIA_TYPES, title="Media Trend Scores Over Time")
    
    # [FIX] 이벤트 마커 표시 (Categorical Axis 호환성 수정)
    # annotation_text를 add_vline에 직접 넣으면 int+str 에러 발생함.
    # 별도의 add_annotation으로 분리하여 처리.
    if enable_maintenance:
        fig_v.add_vline(x="14:00", line_dash="dash", line_color="red")
        fig_v.add_annotation(x="14:00", y=1.05, yref="paper", text="Maintenance", showarrow=False, font=dict(color="red"))
        
    if enable_hottime:
        fig_v.add_vline(x="20:00", line_dash="dash", line_color="gold")
        fig_v.add_annotation(x="20:00", y=1.05, yref="paper", text="Hot Time", showarrow=False, font=dict(color="gold"))
        
    st.plotly_chart(fig_v, use_container_width=True)

    # --- Chart 4: Activity Distribution ---
    st.subheader("🏆 Activity Popularity")
    action_counts = logs['action_counts']
    df_pop = pd.DataFrame({
        "Activity": df_activities['Name'],
        "Category": df_activities['Category'],
        "Count": action_counts
    }).sort_values("Count", ascending=True)
    
    fig_bar = px.bar(df_pop, x="Count", y="Activity", color="Category", orientation='h', title="Total Actions Performed")
    st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("👈 Set simulation parameters and click **Run Simulation** to start.")