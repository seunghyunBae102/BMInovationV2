import numpy as np
import pandas as pd

# ==========================================
# Inference Engine (추론 엔진) v1.1
# ==========================================
# 1. Utility Calculation (효용 계산)
# 2. Decision Making (Knapsack Greedy)
# ==========================================

TAG_LIST = [
    "Competition", "Skill", "Growth", "RPG", "Gambling", "Collection", 
    "Free", "Patience", "Humor", "Trend", "Social", "Info", 
    "Responsibility", "Relax"
]
TAG_TO_IDX = {tag: i for i, tag in enumerate(TAG_LIST)}

def precompute_activity_tags_matrix(df_activities, num_tags=50):
    """
    활동(M개) x 태그(50개) 형태의 One-hot 행렬 생성
    """
    num_acts = len(df_activities)
    act_tag_matrix = np.zeros((num_acts, num_tags))
    
    for i, tags in enumerate(df_activities['Tags']):
        for tag in tags:
            if tag in TAG_TO_IDX:
                idx = TAG_TO_IDX[tag]
                act_tag_matrix[i, idx] = 1.0
                
    return act_tag_matrix

def calculate_utility(agents, df_activities, act_tag_matrix, time_context):
    """
    Utility Matrix [N, M] 계산
    """
    n_agents = len(agents['ids'])
    n_acts = len(df_activities)
    
    # 1. Interest Matching
    interest_scores = np.dot(agents['interests'], act_tag_matrix.T)
    interest_scores = np.maximum(interest_scores, 0.1)

    # 2. Vectors
    base_rewards = df_activities['Base_Reward'].values.reshape(1, -1)
    stress_costs = df_activities['Stress_Cost'].values.reshape(1, -1)
    money_costs = df_activities['Cost'].values.reshape(1, -1)
    
    # 3. Agent States
    current_stress = agents['state_stress']
    loss_aversion = agents['loss_aversion']
    
    # 4. Context Modifier
    stress_mod = time_context['Stress_Mod']
    ad_efficiency = time_context['Ad_Efficiency']
    
    context_reward_mult = np.ones((1, n_acts))
    ad_indices = df_activities.index[df_activities['ID'] == 'ACT_GM_AD'].tolist()
    if ad_indices:
        context_reward_mult[:, ad_indices] = ad_efficiency

    # 5. Calculation
    gains = (base_rewards * context_reward_mult) * (1.0 + interest_scores)
    
    pain_stress = (stress_costs * stress_mod) * (1.0 + (current_stress * 0.01))
    pain_money = money_costs * 0.001 
    
    losses = pain_stress + pain_money
    
    # Prospect Theory: Loss Aversion
    weighted_losses = losses * loss_aversion
    
    utility_matrix = gains - weighted_losses
    
    # 6. Noise
    noise = np.random.normal(0, 2.0, size=(n_agents, n_acts))
    utility_matrix += noise
    
    return utility_matrix

def decide_actions_knapsack(utility_matrix, df_activities, agents):
    """
    [벡터화된 냅색 알고리즘]
    각 에이전트의 Capacity 내에서 가성비(Utility/Intensity)가 좋은 활동을 순서대로 담음.
    
    Returns:
        final_mask (np.ndarray): [N, M] bool matrix (1=수행, 0=미수행)
    """
    n_agents, n_acts = utility_matrix.shape
    agent_caps = agents['attention_cap'] # [N, 1]
    
    # 1. Intensity Vector [1, M]
    intensities = df_activities['Intensity'].values.reshape(1, -1)
    
    # 2. 가성비(Ratio) 계산: Utility / Intensity
    # Intensity가 0인 경우를 대비해 아주 작은 수(1e-6) 더함
    ratios = utility_matrix / (intensities + 1e-6)
    
    # 3. 정렬 (Sorting) - 가성비 높은 순서대로 인덱스 정렬
    # argsort는 오름차순이므로 [:, ::-1]로 뒤집어서 내림차순 정렬
    sorted_indices = np.argsort(ratios, axis=1)[:, ::-1] # [N, M]
    
    # 4. 정렬된 순서대로 Intensity 배열 재배치 (Advanced Indexing)
    # 각 에이전트별로 활동 순서가 다르므로 복잡한 인덱싱 필요
    row_indices = np.arange(n_agents)[:, np.newaxis] # [N, 1]
    sorted_intensities = intensities[0, sorted_indices] # [N, M] (Broadcasting으로 안됨, 아래 방식 사용)
    sorted_intensities = intensities[0][sorted_indices] # [N, M] - 정렬된 순서의 강도들
    
    # 5. 누적 합 (Cumulative Sum) 계산
    # 예: [90, 15, 60] -> [90, 105, 165]
    cum_intensities = np.cumsum(sorted_intensities, axis=1) # [N, M]
    
    # 6. 용량 마스크 생성 (Capacity Check)
    # 누적 합이 내 용량보다 작거나 같은 것만 True
    allowed_mask_sorted = cum_intensities <= agent_caps # [N, M]
    
    # 7. 원래 인덱스로 복원 (Restore Order)
    # 현재 allowed_mask_sorted는 '정렬된 순서' 기준임. 이를 원래 활동 ID 순서로 돌려놔야 함.
    final_mask = np.zeros((n_agents, n_acts), dtype=bool)
    
    # 행(에이전트)별로, 정렬된 인덱스 위치에 마스크 값을 할당
    # 이 부분은 완전 벡터화가 까다롭지만, flat index를 활용하면 가능함
    
    # Flat Index 변환 로직:
    # (Row, Col) -> Row * n_cols + Col
    flat_sorted_indices = row_indices * n_acts + sorted_indices
    final_mask.ravel()[flat_sorted_indices.ravel()] = allowed_mask_sorted.ravel()
    
    return final_mask