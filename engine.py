import numpy as np
import pandas as pd
import inference
import psy_sim_config

# ==========================================
# Simulation Engine v2.2 (Dynamic World) - Hotfix
# ==========================================
# [Update Log]
# - Fixed KeyError: Added missing 'avg_dopamine' and 'avg_anxiety' logs
# - Event System: 특정 틱에 전역 버프/디버프 적용
# - Dynamic Modifiers: 활동의 보상/비용을 실시간으로 조작
# ==========================================

def process_gacha_mechanics(agents, action_mask, df_activities, act_tag_matrix):
    # (기존 v2.1 로직 동일 - 생략 없이 포함)
    n_agents = len(agents['ids'])
    gambling_tag_idx = inference.TAG_TO_IDX.get("Gambling")
    if gambling_tag_idx is None: return
    
    is_gacha_act = act_tag_matrix[:, gambling_tag_idx] > 0
    gacha_actions_mask = action_mask[:, is_gacha_act]
    did_gacha = np.any(gacha_actions_mask, axis=1)
    
    if not np.any(did_gacha): return
    
    base_prob = 0.05
    pity_bonus = agents['gacha_pity_count'][did_gacha] * 0.005
    success_prob = base_prob + pity_bonus
    
    roll = np.random.rand(np.sum(did_gacha), 1)
    is_success = roll < success_prob
    
    success_indices = np.where(did_gacha)[0][is_success.flatten()]
    fail_indices = np.where(did_gacha)[0][~is_success.flatten()]
    
    if len(success_indices) > 0:
        agents['state_dopamine'][success_indices] = 100.0
        agents['state_stress'][success_indices] -= 30.0
        agents['gacha_pity_count'][success_indices] = 0
        agents['recent_fail_streak'][success_indices] = 0
        
    if len(fail_indices) > 0:
        agents['state_stress'][fail_indices] += 20.0
        agents['gacha_pity_count'][fail_indices] += 1
        agents['recent_fail_streak'][fail_indices] += 1
        agents['state_dopamine'][fail_indices] -= 5.0

    agents['state_stress'] = np.clip(agents['state_stress'], 0, 100)
    agents['state_dopamine'] = np.clip(agents['state_dopamine'], 0, 100)


def run_simulation(agents, df_activities, df_time_slots=None, events=None): 
    """
    events: dict { tick: {"Type": str, "Target": str, "Value": float} }
    """
    n_agents = len(agents['ids'])
    n_acts = len(df_activities)
    
    # Data Setup
    expected_cols = ['Fun_Reward', 'Growth_Reward', 'Difficulty']
    if not all(col in df_activities.columns for col in expected_cols):
        if 'Fun_Reward' not in df_activities.columns: df_activities['Fun_Reward'] = df_activities.get('Base_Reward', 0.0)
        if 'Growth_Reward' not in df_activities.columns: df_activities['Growth_Reward'] = 0.0
        if 'Difficulty' not in df_activities.columns: df_activities['Difficulty'] = 0

    act_tag_matrix = inference.precompute_activity_tags_matrix(df_activities)
    act_media_matrix = inference.precompute_media_matrix(df_activities)
    
    # [Base Vectors] 초기값 저장 (이벤트 끝나면 복구용)
    base_vec_fun = df_activities['Fun_Reward'].values.reshape(1, -1)
    base_vec_growth = df_activities['Growth_Reward'].values.reshape(1, -1)
    base_vec_money = df_activities['Cost'].values.reshape(1, -1)
    base_vec_diff = df_activities['Difficulty'].values.reshape(1, -1)
    vec_stress_cost = df_activities['Stress_Cost'].values.reshape(1, -1)

    _, stress_table, ad_eff_table = psy_sim_config.load_life_patterns()
    TOTAL_TICKS = len(stress_table)

    logs = {
        "time": [],
        "total_revenue": [],
        "avg_stress": [],
        "avg_dopamine": [], # [FIX] Added missing key
        "avg_anxiety": [],  # [FIX] Added missing key
        "pattern_stress": {0:[], 1:[], 2:[], 3:[]}, # [FIX] Added missing key
        "action_counts": np.zeros(n_acts),
        "viral_trends": [],
        "events": [] 
    }
    total_revenue = 0
    viral_scores = np.zeros((1, inference.NUM_MEDIA_TYPES))

    print(f"Starting Simulation v2.2 (Dynamic) for {n_agents} agents...")
    
    for tick in range(TOTAL_TICKS):
        hour = (tick * 15) // 60
        
        # ----------------------------------------
        # [NEW] Event Processor
        # ----------------------------------------
        # 매 틱마다 Base Vector로 초기화 (이전 틱 효과 제거)
        current_vec_fun = base_vec_fun.copy()
        current_vec_diff = base_vec_diff.copy()
        
        event_msg = ""
        if events and tick in events:
            evt = events[tick]
            event_msg = f"[EVENT] {evt['Type']} on {evt['Target']}"
            logs['events'].append(f"{hour:02d}:{tick%4*15:02d} - {evt['Type']}")
            
            # Target Media Group 찾기
            target_media = evt.get("Target")
            target_media_idx = inference.MEDIA_TO_IDX.get(target_media)
            
            if target_media_idx is not None:
                # 해당 미디어 그룹에 속한 활동들의 마스크 [1, M]
                is_target_act = act_media_matrix[:, target_media_idx].reshape(1, -1)
                
                # 이벤트 타입별 로직 적용
                if evt['Type'] == 'SERVER_DOWN':
                    # 서버 점검: 해당 미디어 활동의 난이도를 무한대로 높여버림 (접속 불가 유도)
                    # 혹은 재미 보상을 -1000으로 설정
                    current_vec_fun[is_target_act.astype(bool)] = -1000.0
                    
                elif evt['Type'] == 'HOT_TIME':
                    # 핫타임: 재미 보상 2배
                    current_vec_fun[is_target_act.astype(bool)] *= evt.get("Value", 2.0)
                    
                elif evt['Type'] == 'VIRAL_BOOST':
                    # 바이럴 마케팅: 유행 점수 강제 주입
                    viral_scores[0, target_media_idx] += evt.get("Value", 0.5)

        # Context Mapping
        step_stress_mods = stress_table[tick] 
        step_ad_effs = ad_eff_table[tick]
        agent_pattern_ids = agents['life_pattern'].flatten()
        current_agent_stress_mod = step_stress_mods[agent_pattern_ids].reshape(-1, 1)
        current_agent_ad_eff = step_ad_effs[agent_pattern_ids].reshape(-1, 1)
        
        time_context = {
            'Stress_Mod': current_agent_stress_mod, 
            'Ad_Efficiency': current_agent_ad_eff,
            'Hour': hour
        }
        
        # 1. Perception & Decision (Modified Vectors)
        # 임시 수정: inference.py가 df_activities를 참조하므로 값 덮어쓰기
        df_activities['Fun_Reward'] = current_vec_fun.flatten()
        df_activities['Difficulty'] = current_vec_diff.flatten()
        
        utility_matrix = inference.calculate_utility(
            agents, df_activities, act_tag_matrix, act_media_matrix, 
            time_context, viral_scores=viral_scores
        )
        action_mask = inference.decide_actions_knapsack(
            utility_matrix, df_activities, agents
        )
        
        # ----------------------------------------
        # [Gacha & Social Logic] (v2.1과 동일)
        # ----------------------------------------
        process_gacha_mechanics(agents, action_mask, df_activities, act_tag_matrix)
        
        agent_media_participation = np.dot(action_mask.astype(float), act_media_matrix)
        total_traffic = np.sum(agent_media_participation, axis=0).reshape(1, -1)
        traffic_ratio = total_traffic / n_agents
        viral_scores = (viral_scores * 0.95) + (traffic_ratio * 0.2)
        
        # ----------------------------------------
        # State Update
        # ----------------------------------------
        money_spent = (action_mask * base_vec_money).sum(axis=1).reshape(-1, 1) # 비용은 Base 사용
        agents['wallet'] -= money_spent
        total_revenue += np.sum(money_spent[money_spent > 0])
        
        stress_change = (action_mask * vec_stress_cost).sum(axis=1).reshape(-1, 1)
        agents['state_stress'] = np.clip(agents['state_stress'] + stress_change, 0, 100)
        
        # Needs Update (Modified Rewards 적용)
        fun_gained = (action_mask * current_vec_fun).sum(axis=1).reshape(-1, 1)
        growth_gained = (action_mask * base_vec_growth).sum(axis=1).reshape(-1, 1)
        
        agents['state_dopamine'] = np.clip(agents['state_dopamine'] + (fun_gained * 0.2) - 2.0, 0, 100)
        agents['state_anxiety'] = np.clip(agents['state_anxiety'] - (growth_gained * 0.2) + 0.5, 0, 100)

        agent_media_activity = np.dot(action_mask.astype(float), act_media_matrix)
        has_activity = agent_media_activity.sum(axis=1) > 0
        if np.any(has_activity):
            primary_media_indices = np.argmax(agent_media_activity, axis=1)
            agents['state_current_media'][has_activity] = primary_media_indices[has_activity].reshape(-1, 1)

        is_active_media = (agent_media_activity > 0).astype(float)
        agents['media_boredom'] = np.clip(agents['media_boredom'] + (is_active_media * 0.1) - ((1.0 - is_active_media) * 0.05), 0.0, 1.0)

        experienced_tags = np.dot(action_mask.astype(float), act_tag_matrix)
        learning_rate = 0.001
        dynamic_lr = learning_rate * (1.0 + agents['traits_big5'][:, 0].reshape(-1, 1))
        agents['interests'] = np.clip(agents['interests'] + experienced_tags * dynamic_lr, 0.0, 1.0)
        
        # Logs
        logs["time"].append(f"{hour:02d}:{tick%4*15:02d}")
        logs["total_revenue"].append(total_revenue)
        logs["avg_stress"].append(np.mean(agents['state_stress']))
        logs["avg_dopamine"].append(np.mean(agents['state_dopamine'])) # [FIX] Added
        logs["avg_anxiety"].append(np.mean(agents['state_anxiety']))   # [FIX] Added
        logs["action_counts"] += action_mask.sum(axis=0)
        logs["viral_trends"].append(viral_scores.flatten().copy())

        # [FIX] Pattern Stress Logging
        for pid in range(4):
            mask = (agents['life_pattern'] == pid).flatten()
            if np.any(mask):
                avg_s = np.mean(agents['state_stress'][mask])
                logs["pattern_stress"][pid].append(avg_s)
            else:
                logs["pattern_stress"][pid].append(0)
        
        if tick % 16 == 0:
            extra_info = f" | {event_msg}" if event_msg else ""
            print(f"[{logs['time'][-1]}] Rev: {total_revenue:,.0f}{extra_info}")

    # Restore DF (Clean up)
    df_activities['Fun_Reward'] = base_vec_fun.flatten()
    df_activities['Difficulty'] = base_vec_diff.flatten()

    print("Simulation v2.2 (Dynamic) Complete.")
    return logs