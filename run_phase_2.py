import pandas as pd
import numpy as np
import genesis
import psy_sim_config
import inference
import time

def main():
    print("=== [Phase 2] Utility Calculation Engine Test ===\n")

    # 1. 데이터 로드 (Phase 1)
    print("1. Loading Data...")
    df_activities = psy_sim_config.load_activity_table()
    df_time_slots = psy_sim_config.load_time_slots()
    
    # 2. 태그 매트릭스 전처리 (Activity x Tag)
    # 시뮬레이션 시작 전 한 번만 수행
    print("2. Precomputing Activity-Tag Matrix...")
    act_tag_matrix = inference.precompute_activity_tags_matrix(df_activities)
    print(f"-> Shape: {act_tag_matrix.shape} (Activities x Tags)")
    
    # 3. 에이전트 생성
    print("3. Creating Agents...")
    N_AGENTS = 10000
    population = genesis.create_agent_population(N_AGENTS)
    
    # 4. 효용 계산 테스트
    print("\n4. Running Utility Calculation (Vectorized)...")
    
    # 테스트 시나리오: [08:00 출근길] 상황 가정
    # 출근길(COMMUTE_AM)은 스트레스 민감도(Stress_Mod)가 1.5배 높음
    current_time_idx = 32 # 08:00
    time_context = df_time_slots.iloc[current_time_idx]
    print(f"-> Context: {time_context['Hour']}:00 ({time_context['Context']})")
    print(f"-> Stress Mod: {time_context['Stress_Mod']}, Ad Eff: {time_context['Ad_Efficiency']}")
    
    start_time = time.time()
    
    # 핵심 함수 호출
    utility_matrix = inference.calculate_utility(
        population, 
        df_activities, 
        act_tag_matrix, 
        time_context
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n-> Calculation Complete!")
    print(f"-> Time Elapsed: {elapsed:.4f} sec for {N_AGENTS} agents.")
    print(f"-> Utility Matrix Shape: {utility_matrix.shape}")
    
    # 5. 결과 검증 (샘플 에이전트의 선호도 분석)
    # 0번 에이전트가 이 상황에서 가장 하고 싶어하는 활동 Top 3
    agent_idx = 0
    agent_utils = utility_matrix[agent_idx]
    
    # 효용이 높은 순으로 정렬 (내림차순)
    top_indices = np.argsort(agent_utils)[::-1][:3]
    
    print(f"\n[Analysis] Agent #{agent_idx}'s Top 3 Desired Actions at {time_context['Hour']}:00")
    print(f"- Agent Traits: Big5 {population['traits_big5'][agent_idx].round(2)}")
    print("-" * 50)
    for idx in top_indices:
        act_row = df_activities.iloc[idx]
        score = agent_utils[idx]
        print(f"Rank {list(top_indices).index(idx)+1}: [{act_row['Name']}] (Util: {score:.2f})")
        print(f"   -> Category: {act_row['Category']}, Tags: {act_row['Tags']}")
    print("-" * 50)
    
    # 전체 통계: 현재 가장 인기 있는 활동은?
    # 각 에이전트별 최선호 활동(argmax)을 뽑아서 카운트
    best_choices = np.argmax(utility_matrix, axis=1)
    print(f"\n[Global Statistics] Most Popular Actions (Top 1 choices by 10,000 agents)")
    from collections import Counter
    counts = Counter(best_choices)
    
    for idx, count in counts.most_common(5):
        act_name = df_activities.iloc[idx]['Name']
        print(f"- {act_name}: {count} agents ({count/N_AGENTS*100:.1f}%)")

if __name__ == "__main__":
    main()