import numpy as np
import pandas as pd
import inference

# ==========================================
# Simulation Engine (메인 엔진)
# ==========================================
# 시간(Tick)을 진행시키며 에이전트들의 상태를
# 일괄 업데이트하는 로직을 담당합니다.
# ==========================================

def run_simulation(agents, df_activities, df_time_slots):
    """
    하루(96 ticks) 동안 시뮬레이션을 수행하고 로그를 반환합니다.
    """
    n_agents = len(agents['ids'])
    n_acts = len(df_activities)
    
    # 태그 매트릭스 미리 계산
    act_tag_matrix = inference.precompute_activity_tags_matrix(df_activities)
    
    # 벡터화 연산을 위한 Cost/Reward 벡터 준비
    # shape: [1, M]
    vec_money_cost = df_activities['Cost'].values.reshape(1, -1)
    vec_stress_cost = df_activities['Stress_Cost'].values.reshape(1, -1)
    
    # 통계 수집용 로그
    logs = {
        "time": [],
        "total_revenue": [],
        "avg_stress": [],
        "active_user_count": [], # 아무것도 안함(REST 포함) 제외한 활동 유저 수
        "action_counts": np.zeros(n_acts) # 활동별 누적 수행 횟수
    }
    
    # 누적 매출 (Revenue)
    total_revenue = 0
    
    print(f"Starting Simulation for {n_agents} agents (96 Ticks)...")
    
    # === Main Loop (00:00 ~ 23:45) ===
    for tick in range(len(df_time_slots)):
        time_ctx = df_time_slots.iloc[tick]
        
        # 1. Perception & Decision (인지 및 판단)
        # ----------------------------------------
        utility_matrix = inference.calculate_utility(
            agents, df_activities, act_tag_matrix, time_ctx
        )
        action_mask = inference.decide_actions_knapsack(
            utility_matrix, df_activities, agents
        )
        
        # 2. State Update (상태 업데이트) - Vectorized
        # ----------------------------------------
        # action_mask: [N, M] bool
        
        # (1) 지갑 업데이트 (비용 차감)
        # 수행한 활동의 비용 합계 계산: (Mask * Cost).sum(axis=1)
        money_spent = (action_mask * vec_money_cost).sum(axis=1).reshape(-1, 1)
        agents['wallet'] -= money_spent
        
        # 매출 집계 (음수 비용은 소득이므로 제외, 양수 비용만 매출로 잡음)
        # 여기서는 게임사의 매출을 잡아야 하므로, 'Category'가 'GAME'인 것의 비용만 합산
        # (간단히 하기 위해 money_spent 중 양수만 합산)
        step_revenue = np.sum(money_spent[money_spent > 0])
        total_revenue += step_revenue
        
        # (2) 스트레스/피로도 업데이트
        # 스트레스 변화량 계산
        stress_change = (action_mask * vec_stress_cost).sum(axis=1).reshape(-1, 1)
        agents['state_stress'] += stress_change
        
        # 스트레스는 0~100 사이로 유지 (Clipping)
        agents['state_stress'] = np.clip(agents['state_stress'], 0, 100)
        
        # 3. Neuroplasticity (취향 학습)
        # ----------------------------------------
        # 경험한 활동의 태그에 대해 선호도가 조금씩 강화됨 (Hebbian Learning)
        # Logic: 내_취향 += 학습률 * (수행한_활동들 @ 활동_태그_매트릭스)
        
        # 이번 틱에 수행한 활동들이 가진 태그들의 합 [N, 50]
        experienced_tags = np.dot(action_mask.astype(float), act_tag_matrix)
        
        # 학습률 (Learning Rate): 0.001 (천천히 변함)
        learning_rate = 0.001
        
        # 개방성(Openness)이 높은 에이전트는 더 빨리 변함
        # traits_big5[:, 0] -> Openness
        openness = agents['traits_big5'][:, 0].reshape(-1, 1)
        dynamic_lr = learning_rate * (1.0 + openness) # 0.001 ~ 0.002
        
        # 업데이트
        agents['interests'] += experienced_tags * dynamic_lr
        agents['interests'] = np.clip(agents['interests'], 0.0, 1.0)
        
        # 4. Logging (로그 기록)
        # ----------------------------------------
        logs["time"].append(f"{time_ctx['Hour']:02d}:{time_ctx['Time_Index']%4*15:02d}")
        logs["total_revenue"].append(total_revenue)
        logs["avg_stress"].append(np.mean(agents['state_stress']))
        
        # 활동 카운트 집계
        logs["action_counts"] += action_mask.sum(axis=0)
        
        # 진행 상황 출력 (4시간마다)
        if tick % 16 == 0:
            print(f"[{logs['time'][-1]}] Rev: {total_revenue:,.0f} | Stress: {logs['avg_stress'][-1]:.1f}")

    print("Simulation Complete.")
    return logs