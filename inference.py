import numpy as np
import pandas as pd

# ==========================================
# Inference Engine v2.3 (Tuning)
# ==========================================
# [Update Log]
# - Algorithmic Resistance: VIDEO 매체는 지루함 페널티 감소 (알고리즘 효과)
# ==========================================

TAG_LIST = [
    "Competition", "Skill", "Growth", "RPG", "Gambling", "Collection", 
    "Free", "Patience", "Humor", "Trend", "Social", "Info", 
    "Responsibility", "Relax", "Strategy", "KillingTime", "Story", 
    "Future", "Knowledge"
]
TAG_TO_IDX = {tag: i for i, tag in enumerate(TAG_LIST)}

MEDIA_TYPES = ["GAME", "VIDEO", "BOOK", "WORK", "COMM", "LIFE"]
MEDIA_TO_IDX = {m: i for i, m in enumerate(MEDIA_TYPES)}
NUM_MEDIA_TYPES = len(MEDIA_TYPES)

def precompute_activity_tags_matrix(df_activities, num_tags=50):
    num_acts = len(df_activities)
    act_tag_matrix = np.zeros((num_acts, num_tags))
    for i, tags in enumerate(df_activities['Tags']):
        if isinstance(tags, str): tag_list = tags.split('|')
        else: tag_list = tags if isinstance(tags, list) else []
        for tag in tag_list:
            if tag in TAG_TO_IDX: act_tag_matrix[i, TAG_TO_IDX[tag]] = 1.0
    return act_tag_matrix

def precompute_media_matrix(df_activities):
    num_acts = len(df_activities)
    act_media_matrix = np.zeros((num_acts, NUM_MEDIA_TYPES))
    if 'Media_Group' not in df_activities.columns: return act_media_matrix
    for i, media_group in enumerate(df_activities['Media_Group']):
        if media_group in MEDIA_TO_IDX:
            act_media_matrix[i, MEDIA_TO_IDX[media_group]] = 1.0
    return act_media_matrix

def calculate_utility(agents, df_activities, act_tag_matrix, act_media_matrix, time_context, viral_scores=None):
    n_agents = len(agents['ids'])
    n_acts = len(df_activities)
    
    # --- Data Extraction ---
    vec_fun = df_activities.get('Fun_Reward', pd.Series(0)).values.reshape(1, -1)
    vec_growth = df_activities.get('Growth_Reward', pd.Series(0)).values.reshape(1, -1)
    vec_diff = df_activities.get('Difficulty', pd.Series(0)).values.reshape(1, -1)
    vec_stress_cost = df_activities['Stress_Cost'].values.reshape(1, -1)
    vec_money_cost = df_activities['Cost'].values.reshape(1, -1)
    
    state_dopamine = agents['state_dopamine']
    state_anxiety = agents['state_anxiety']
    state_stress = agents['state_stress']
    state_current_media = agents['state_current_media']
    traits_intel = agents['traits_intel']
    loss_aversion = agents['loss_aversion']
    traits_extraversion = agents['traits_big5'][:, 2].reshape(-1, 1)

    # 1. Needs Weighting
    w_fun = np.clip((100.0 - state_dopamine) / 100.0, 0.1, 2.0)
    w_growth = 1.0 + (state_anxiety / 20.0)
    base_utility = (vec_fun * w_fun) + (vec_growth * w_growth)
    
    interest_scores = np.dot(agents['interests'], act_tag_matrix.T)
    base_utility *= (1.0 + interest_scores)

    # 2. Difficulty Penalty
    diff_gap = vec_diff - traits_intel
    penalty_flow = np.maximum(diff_gap, 0) * 1.5 
    
    # 3. Inertia Bonus
    agent_media_onehot = np.zeros((n_agents, NUM_MEDIA_TYPES))
    valid_mask = (state_current_media >= 0).flatten()
    if np.any(valid_mask):
        valid_indices = state_current_media[valid_mask].flatten()
        agent_media_onehot[valid_mask, valid_indices] = 1.0
    inertia_matrix = np.dot(agent_media_onehot, act_media_matrix.T)
    inertia_bonus = inertia_matrix * 10.0

    # 4. Media Saturation (Tuned for Short-form)
    # 일반적인 물림 페널티
    saturation_penalty = np.dot(agents['media_boredom'], act_media_matrix.T) * 2.0
    
    # [FIX] VIDEO(Short-form) 계열은 알고리즘 추천으로 인해 지루함이 덜함
    # act_media_matrix에서 VIDEO 컬럼(idx 1)이 1인 활동들 찾기
    video_idx = MEDIA_TO_IDX.get("VIDEO", 1)
    is_video_act = act_media_matrix[:, video_idx].reshape(1, -1) # [1, M]
    
    # 비디오 활동에 대해서는 페널티를 50%만 적용
    saturation_penalty = np.where(is_video_act > 0, saturation_penalty * 0.5, saturation_penalty)

    # 5. Social Bonus
    social_bonus = 0
    if viral_scores is not None:
        media_viral_val = np.dot(viral_scores, act_media_matrix.T) 
        social_bonus = media_viral_val * (traits_extraversion * 5.0)
        
    # [Rage Bet Bonus]
    gambler_fallacy = agents['gambler_fallacy']
    fail_streak = agents['recent_fail_streak']
    gambling_tag_idx = TAG_TO_IDX.get("Gambling", -1)
    rage_bonus = np.zeros((n_agents, n_acts))
    if gambling_tag_idx != -1:
        is_gambling_act = act_tag_matrix[:, gambling_tag_idx].reshape(1, -1)
        rage_factor = (fail_streak * gambler_fallacy * 50.0)
        rage_bonus = rage_factor * is_gambling_act

    # 6. Cost & Context
    stress_mod = time_context['Stress_Mod']
    pain_stress = (vec_stress_cost * stress_mod) * (1.0 + (state_stress * 0.01))
    pain_money = vec_money_cost * 0.001 
    total_pain = (pain_stress + pain_money) * loss_aversion

    # Final Aggregation
    utility_matrix = base_utility + inertia_bonus + social_bonus + rage_bonus - penalty_flow - saturation_penalty - total_pain
    
    noise = np.random.normal(0, 2.0, size=(n_agents, n_acts))
    utility_matrix += noise
    
    return utility_matrix

def decide_actions_knapsack(utility_matrix, df_activities, agents):
    # (기존 Knapsack 로직 동일)
    n_agents, n_acts = utility_matrix.shape
    agent_caps = agents['attention_cap']
    intensities = df_activities['Intensity'].values.reshape(1, -1)
    safe_intensities = intensities.copy()
    safe_intensities[safe_intensities == 0] = 0.1
    ratios = utility_matrix / safe_intensities
    
    sorted_indices = np.argsort(ratios, axis=1)[:, ::-1]
    row_indices = np.arange(n_agents)[:, np.newaxis]
    
    sorted_intensities = intensities[0][sorted_indices]
    cum_intensities = np.cumsum(sorted_intensities, axis=1)
    
    allowed_mask_sorted = cum_intensities <= agent_caps
    
    final_mask = np.zeros((n_agents, n_acts), dtype=bool)
    flat_sorted_indices = row_indices * n_acts + sorted_indices
    final_mask.ravel()[flat_sorted_indices.ravel()] = allowed_mask_sorted.ravel()
    
    return final_mask