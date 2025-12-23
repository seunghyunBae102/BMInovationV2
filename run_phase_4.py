import pandas as pd
import numpy as np
import genesis
import psy_sim_config
import engine
import time
import matplotlib.pyplot as plt

def main():
    print("=== [Phase 4] Full Day Simulation Test ===\n")
    
    # 1. Init
    print("1. Loading Data & Creating Agents...")
    df_activities = psy_sim_config.load_activity_table()
    df_time_slots = psy_sim_config.load_time_slots()
    
    N_AGENTS = 10000
    population = genesis.create_agent_population(N_AGENTS)
    
    # 2. Run Engine
    print("\n2. Running Simulation Engine (24h)...")
    start_time = time.time()
    
    logs = engine.run_simulation(population, df_activities, df_time_slots)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n-> Simulation Finished in {elapsed:.4f} seconds.")
    
    # 3. Summary Report
    print("\n=== [Daily Report] ===")
    print(f"Total Agents: {N_AGENTS}")
    print(f"Total Revenue: {logs['total_revenue'][-1]:,.0f} Gold")
    print(f"Final Avg Stress: {logs['avg_stress'][-1]:.2f} / 100")
    
    # Top 3 Popular Activities
    print("\n[Top 3 Activities by Frequency]")
    total_actions = logs['action_counts']
    top_indices = np.argsort(total_actions)[::-1][:3]
    
    for rank, idx in enumerate(top_indices):
        act_name = df_activities.iloc[idx]['Name']
        count = int(total_actions[idx])
        print(f"{rank+1}. {act_name}: {count:,} times executed")

    # 4. Simple Visualization (Console based Mock-up)
    # 실제 그래프는 Phase 5(Streamlit)에서 그리지만, 여기선 데이터 흐름 확인용
    print("\n[Stress Trend (00:00 -> 23:45)]")
    # 4시간 간격으로 스트레스 수치 출력
    for i in range(0, 96, 16):
        t = logs['time'][i]
        s = logs['avg_stress'][i]
        bar = "#" * int(s // 5)
        print(f"{t} | {s:5.1f} | {bar}")

if __name__ == "__main__":
    main()