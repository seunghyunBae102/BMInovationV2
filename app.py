import streamlit as st
import pandas as pd
import numpy as np
import time

# 기존 모듈 임포트
import genesis
import psy_sim_config
import engine
import create_csv_data
import os

# ==========================================
# Streamlit Dashboard Configuration
# ==========================================
st.set_page_config(
    page_title="Psy-Sim: Market Simulator",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# Helper Functions
# ==========================================
@st.cache_data
def load_configs():
    # 데이터 폴더가 없으면 생성
    if not os.path.exists('data/activities.csv'):
        create_csv_data.create_initial_csvs()
        
    act = psy_sim_config.load_activity_table()
    time_slots = psy_sim_config.load_time_slots()
    return act, time_slots

# ==========================================
# Sidebar: Simulation Parameters
# ==========================================
st.sidebar.title("🎮 Psy-Sim Controller")
st.sidebar.markdown("---")

# 1. 에이전트 설정
st.sidebar.subheader("Population Settings")
n_agents = st.sidebar.slider("Number of Agents", min_value=100, max_value=20000, value=1000, step=100)

# 2. 경제 설정 (추가 파라미터 예시)
st.sidebar.subheader("Economy Settings")
# 간단한 밸런스 조절을 위한 계수 (실제 엔진 연동은 추후 확장 가능)
ad_revenue_mult = st.sidebar.slider("Ad Revenue Multiplier", 0.5, 2.0, 1.0)

# 3. 실행 버튼
st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Simulation", type="primary")

st.sidebar.markdown("""
### ℹ️ About
**Psy-Sim v1.5**
- **Engine:** Vectorized NumPy
- **Logic:** Knapsack + Needs(Fun/Growth) + Inertia
- **Context:** Attention Economy
""")

# ==========================================
# Main Page
# ==========================================
st.title("🧠 Psy-Sim: Psychological Market Simulator")
st.markdown(f"Simulating **{n_agents:,}** unique personas based on Big5 traits & Attention Economy.")

# Config 로드 및 표시
df_activities, df_time_slots = load_configs()

with st.expander("📊 View Simulation Rules (Data Config)"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("Reference: Activity Table")
        
        # [Fix] 컬럼 유무 확인하여 동적으로 표시 (v1.0 / v1.5 호환)
        display_cols = ['ID', 'Name', 'Category', 'Intensity']
        
        # v1.0 호환
        if 'Base_Reward' in df_activities.columns:
            display_cols.append('Base_Reward')
        
        # v1.5 호환
        if 'Fun_Reward' in df_activities.columns:
            display_cols.append('Fun_Reward')
        if 'Growth_Reward' in df_activities.columns:
            display_cols.append('Growth_Reward')
        if 'Difficulty' in df_activities.columns:
            display_cols.append('Difficulty')

        st.dataframe(df_activities[display_cols])
        
    with col2:
        st.write("Reference: Time Slots (Sample)")
        st.dataframe(df_time_slots.head(10))

# ==========================================
# Simulation Logic
# ==========================================
if run_btn:
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 1. Genesis (에이전트 생성)
    status_text.write("🧬 Generating Synthetic Population...")
    population = genesis.create_agent_population(n_agents)
    progress_bar.progress(20)
    
    time.sleep(0.5) # UX를 위한 짧은 대기
    
    # 2. Run Engine (시뮬레이션)
    status_text.write("⚙️ Running Physics Engine (24h Loop)...")
    
    # 엔진 실행 (Phase 4의 함수 사용)
    # population 딕셔너리는 Mutable이므로 내부 값이 계속 업데이트됨
    start_time = time.time()
    logs = engine.run_simulation(population, df_activities, df_time_slots)
    end_time = time.time()
    
    progress_bar.progress(100)
    status_text.success(f"✅ Simulation Complete in {end_time - start_time:.4f} seconds!")
    
    # ==========================================
    # Results Visualization
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Simulation Report")

    # 1. Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_revenue = logs['total_revenue'][-1]
    avg_stress = logs['avg_stress'][-1]
    
    col1.metric("Total Revenue", f"{total_revenue:,.0f} G", delta="Daily Gross")
    col2.metric("Avg Stress", f"{avg_stress:.1f} / 100", delta_color="inverse")
    col3.metric("Simulated Time", "24 Hours")
    col4.metric("Agents", f"{n_agents:,}")

    # 2. Time Series Analysis (Line Chart)
    st.subheader("⏱️ 24h Trends: Stress vs Revenue")
    
    # 로그 데이터를 DataFrame으로 변환
    df_logs = pd.DataFrame({
        "Time": logs['time'],
        "Cumulative Revenue": logs['total_revenue'],
        "Average Stress": logs['avg_stress']
    })
    
    # 스트레스와 매출 그래프 그리기
    chart_data = df_logs.set_index("Time")[["Average Stress"]]
    st.line_chart(chart_data, color="#FF4B4B") # Red for Stress
    
    st.caption("Cumulative Revenue Growth")
    st.area_chart(df_logs.set_index("Time")[["Cumulative Revenue"]], color="#29B5E8")
    
    # [NEW] Needs Analysis (v1.5)
    if 'avg_dopamine' in logs:
        st.subheader("🧠 Psychological Needs Trends")
        st.write("도파민(재미) vs 불안(성장)의 하루 변화")
        df_needs = pd.DataFrame({
            "Time": logs['time'],
            "Avg Dopamine": logs['avg_dopamine'],
            "Avg Anxiety": logs['avg_anxiety']
        })
        st.line_chart(df_needs.set_index("Time"))

    # 3. Activity Popularity (Bar Chart)
    st.subheader("🏆 Most Popular Activities")
    
    # 활동별 카운트 매핑
    action_counts = logs['action_counts']
    df_popularity = pd.DataFrame({
        "Activity": df_activities['Name'],
        "Category": df_activities['Category'],
        "Count": action_counts
    }).sort_values("Count", ascending=False)
    
    st.bar_chart(df_popularity.set_index("Activity")["Count"])

    # 4. Micro Analysis: Trait vs Result (Scatter Plot)
    st.subheader("🔬 Micro Analysis: Personality vs Wallet")
    
    # Scatter Plot을 위한 데이터프레임 생성
    sample_size = min(n_agents, 1000)
    indices = np.random.choice(n_agents, sample_size, replace=False)
    
    df_micro = pd.DataFrame({
        "Conscientiousness": population['traits_big5'][indices, 1],
        "Openness": population['traits_big5'][indices, 0],
        "Final Wallet": population['wallet'][indices].flatten(),
        "Stress Level": population['state_stress'][indices].flatten()
    })
    
    st.scatter_chart(
        df_micro,
        x="Conscientiousness",
        y="Final Wallet",
        color="Stress Level",
        size="Openness",
        use_container_width=True
    )
    
    st.info("💡 Tip: 점의 색깔은 스트레스 수치, 크기는 개방성(Openness)을 의미합니다.")

else:
    st.info("👈 Please set parameters in the sidebar and click 'Run Simulation'")