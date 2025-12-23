import numpy as np
import pandas as pd
import inference

# ==========================================
# Simulation Engine v1.5
# ==========================================
# [Update Log]
# - 매체 상태(Media State) 추적: 관성 및 물림(Boredom) 업데이트
# - 심리 상태(Needs) 업데이트: 도파민(재미) vs 불안(성장)
# - 벡터화된 상태 전이 로직 적용
# ==========================================

def run_simulation(agents, df_activities, df_time_slots):
    """
    하루(96 ticks) 동안 시뮬레이션을 수행하고 로그를 반환합니다.
    """
    n_agents = len(agents['ids'])
    n_acts = len(df_activities)
    
    # ----------------------------------------
    # [Fix] Data Schema Compatibility Patch
    # ----------------------------------------
    # v1.0 데이터(CSV)를 v1.5 코드로 돌릴 때 컬럼 부재로 인한 차원 에러 방지
    # Fun_Reward 등이 없으면 Base_Reward나 0.0으로 초기화하여 (1, M) 차원 보장
    expected_cols = ['Fun_Reward', 'Growth_Reward', 'Difficulty']
    if not all(col in df_activities.columns for col in expected_cols):
        print("[Warning] Data Schema mismatch. Applying v1.0 -> v1.5 compatibility defaults.")
        if 'Fun_Reward' not in df_activities.columns:
            # 기존 Base_Reward가 있으면 사용, 없으면 0
            df_activities['Fun_Reward'] = df_activities.get('Base_Reward', 0.0)
        if 'Growth_Reward' not in df_activities.columns:
            df_activities['Growth_Reward'] = 0.0
        if 'Difficulty' not in df_activities.columns:
            df_activities['Difficulty'] = 0

    # ----------------------------------------
    # 0. Pre-computation (벡터화 준비)
    # ----------------------------------------
    # 태그 및 매체 매트릭스 계산
    act_tag_matrix = inference.precompute_activity_tags_matrix(df_activities)
    act_media_matrix = inference.precompute_media_matrix(df_activities) # [M, K]
    
    # Reward/Cost 벡터 [1, M]
    # 위에서 컬럼 존재를 보장했으므로 .values 사용 시 (M,) -> (1, M) 정상 변환됨
    vec_fun = df_activities['Fun_Reward'].values.reshape(1, -1)
    vec_growth = df_activities['Growth_Reward'].values.reshape(1, -1)
    vec_money_cost = df_activities['Cost'].values.reshape(1, -1)
    vec_stress_cost = df_activities['Stress_Cost'].values.reshape(1, -1)
    
    # 로그 데이터 초기화
    logs = {
        "time": [],
        "total_revenue": [],
        "avg_stress": [],
        "avg_dopamine": [], # [NEW]
        "avg_anxiety": [],  # [NEW]
        "action_counts": np.zeros(n_acts)
    }
    
    total_revenue = 0
    
    print(f"Starting Simulation v1.5 for {n_agents} agents...")
    
    # === Main Loop (00:00 ~ 23:45) ===
    for tick in range(len(df_time_slots)):
        time_ctx = df_time_slots.iloc[tick]
        
        # ----------------------------------------
        # 1. Perception & Decision (인지 및 판단)
        # ----------------------------------------
        # v1.5 로직: Needs, Inertia, Saturation 반영된 효용 계산
        utility_matrix = inference.calculate_utility(
            agents, df_activities, act_tag_matrix, act_media_matrix, time_ctx
        )
        
        # 행동 결정 (Knapsack Greedy)
        action_mask = inference.decide_actions_knapsack(
            utility_matrix, df_activities, agents
        ) # [N, M] bool
        
        # ----------------------------------------
        # 2. Basic State Update (기존 자원)
        # ----------------------------------------
        # 지갑 업데이트
        money_spent = (action_mask * vec_money_cost).sum(axis=1).reshape(-1, 1)
        agents['wallet'] -= money_spent
        
        # 매출 집계 (양수 비용만)
        step_revenue = np.sum(money_spent[money_spent > 0])
        total_revenue += step_revenue
        
        # 스트레스 업데이트
        stress_change = (action_mask * vec_stress_cost).sum(axis=1).reshape(-1, 1)
        agents['state_stress'] += stress_change
        agents['state_stress'] = np.clip(agents['state_stress'], 0, 100)
        
        # ----------------------------------------
        # 3. Advanced State Update (v1.5 Core)
        # ----------------------------------------
        
        # (A) Needs Update (도파민 & 불안)
        # -------------------------------
        # 수행한 활동의 Fun/Growth 총량 계산
        fun_gained = (action_mask * vec_fun).sum(axis=1).reshape(-1, 1)
        growth_gained = (action_mask * vec_growth).sum(axis=1).reshape(-1, 1)
        
        # 도파민: 재미있는 걸 하면 충전됨, 매 틱마다 자연 소모(갈망)
        agents['state_dopamine'] += (fun_gained * 0.2) # 충전
        agents['state_dopamine'] -= 2.0                # 자연 소모 (갈증)
        agents['state_dopamine'] = np.clip(agents['state_dopamine'], 0, 100)
        
        # 불안: 성장 활동을 하면 해소됨, 매 틱마다 자연 증가(미래에 대한 불안)
        agents['state_anxiety'] -= (growth_gained * 0.2) # 해소
        agents['state_anxiety'] += 0.5                   # 자연 증가
        agents['state_anxiety'] = np.clip(agents['state_anxiety'], 0, 100)

        # (B) Media State Update (관성 & 물림)
        # -----------------------------------
        # [N, M] @ [M, K] = [N, K] (에이전트별 매체 참여 여부)
        # 각 에이전트가 이번 틱에 어떤 매체 그룹 활동을 했는지 확인
        agent_media_activity = np.dot(action_mask.astype(float), act_media_matrix)
        
        # 1. Update Current Media (Inertia)
        # 가장 많이 활동한 매체 인덱스를 현재 매체로 설정 (활동 없으면 유지 or -1)
        # 활동이 있었던 에이전트만 업데이트
        has_activity = agent_media_activity.sum(axis=1) > 0
        if np.any(has_activity):
            # argmax로 주된 매체 찾기
            primary_media_indices = np.argmax(agent_media_activity, axis=1)
            # 마스킹하여 업데이트 (활동 안 한 애들은 기존 매체 유지 - 관성 유지)
            # 또는 활동 안 하면 관성이 깨진다고 볼 수도 있음 -> 여기선 유지로 구현
            agents['state_current_media'][has_activity] = primary_media_indices[has_activity].reshape(-1, 1)

        # 2. Update Media Boredom (Saturation)
        # 활동한 매체는 지루함 증가 (+0.1), 안 한 매체는 회복 (-0.05)
        is_active_media = (agent_media_activity > 0).astype(float)
        
        boredom_increase = is_active_media * 0.1
        boredom_recovery = (1.0 - is_active_media) * 0.05
        
        agents['media_boredom'] += boredom_increase
        agents['media_boredom'] -= boredom_recovery
        agents['media_boredom'] = np.clip(agents['media_boredom'], 0.0, 1.0) # 0~1 사이 유지

        # ----------------------------------------
        # 4. Neuroplasticity (취향 학습)
        # ----------------------------------------
        experienced_tags = np.dot(action_mask.astype(float), act_tag_matrix)
        learning_rate = 0.001
        openness = agents['traits_big5'][:, 0].reshape(-1, 1)
        dynamic_lr = learning_rate * (1.0 + openness)
        
        agents['interests'] += experienced_tags * dynamic_lr
        agents['interests'] = np.clip(agents['interests'], 0.0, 1.0)
        
        # ----------------------------------------
        # 5. Logging
        # ----------------------------------------
        logs["time"].append(f"{time_ctx['Hour']:02d}:{time_ctx['Time_Index']%4*15:02d}")
        logs["total_revenue"].append(total_revenue)
        logs["avg_stress"].append(np.mean(agents['state_stress']))
        logs["avg_dopamine"].append(np.mean(agents['state_dopamine'])) # [NEW]
        logs["avg_anxiety"].append(np.mean(agents['state_anxiety']))   # [NEW]
        logs["action_counts"] += action_mask.sum(axis=0)
        
        if tick % 16 == 0:
            print(f"[{logs['time'][-1]}] Rev: {total_revenue:,.0f} | "
                  f"Stress: {logs['avg_stress'][-1]:.1f} | "
                  f"Dopa: {logs['avg_dopamine'][-1]:.1f}")

    print("Simulation v1.5 Complete.")
    return logs