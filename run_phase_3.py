import pandas as pd
import numpy as np
import genesis
import psy_sim_config
import inference
import time

def main():
    print("=== [Phase 3] Decision Making (Knapsack) Test ===\n")

    # 1. Init
    df_activities = psy_sim_config.load_activity_table()
    df_time_slots = psy_sim_config.load_time_slots()
    act_tag_matrix = inference.precompute_activity_tags_matrix(df_activities)
    
    N_AGENTS = 10000
    population = genesis.create_agent_population(N_AGENTS)
    
    # 2. Setup Context (19:00 퇴근길/휴식 시간)
    # 퇴근 후라 Capacity 여유가 좀 있고 스트레스가 풀리는 상황 가정
    time_ctx = df_time_slots.iloc[76] # 19:00
    print(f"Current Context: {time_ctx['Hour']}:00 ({time_ctx['Context']})")
    
    # 3. Calculate Utility
    print("Calculating Utility...")
    utility_matrix = inference.calculate_utility(population, df_activities, act_tag_matrix, time_ctx)
    
    # 4. Decide Actions (Knapsack)
    print("Running Decision Making (Knapsack)...")
    start_time = time.time()
    
    action_mask = inference.decide_actions_knapsack(utility_matrix, df_activities, population)
    
    end_time = time.time()
    print(f"-> Decision Complete in {end_time - start_time:.4f} seconds.")
    print(f"-> Action Mask Shape: {action_mask.shape}")
    
    # 5. Analysis (검증)
    print("\n[Analysis] Multi-Tasking Capability Verification")
    
    # 에이전트별 수행 활동 개수 카운트
    act_counts = np.sum(action_mask, axis=1)
    
    # 멀티태스킹(2개 이상 활동) 성공한 에이전트 수
    multi_taskers = np.sum(act_counts >= 2)
    print(f"- Agents doing 0 actions: {np.sum(act_counts == 0)}")
    print(f"- Agents doing 1 action:  {np.sum(act_counts == 1)}")
    print(f"- Agents doing 2+ actions (Multi-tasking): {multi_taskers} ({multi_taskers/N_AGENTS*100:.1f}%)")
    
    # 샘플 에이전트 상세 로그
    agent_idx = 0
    print(f"\n[Sample Agent #{agent_idx} Log]")
    print(f"- Attention Capacity: {population['attention_cap'][agent_idx][0]}")
    
    # 수행한 활동 목록
    done_indices = np.where(action_mask[agent_idx])[0]
    total_intensity = 0
    total_utility = 0
    
    print("- Selected Actions:")
    for idx in done_indices:
        act = df_activities.iloc[idx]
        util = utility_matrix[agent_idx, idx]
        print(f"  [{act['Name']}] (Intensity: {act['Intensity']}, Utility: {util:.2f})")
        total_intensity += act['Intensity']
        total_utility += util
        
    print(f"- Total Intensity Used: {total_intensity} / {population['attention_cap'][agent_idx][0]}")
    print(f"- Total Utility Gained: {total_utility:.2f}")

    # 가성비(Ratio) 검증: 선택되지 않은 활동 중 가성비가 더 좋았는데 용량 때문에 탈락한 게 있는지?
    # (Greedy 특성상 있을 수 있음, 하지만 용량 초과여야 함)
    
if __name__ == "__main__":
    main()