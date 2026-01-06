import numpy as np
import pandas as pd

# ==========================================
# Genesis Module v2.1
# ==========================================
# [Update Log]
# - Gacha State: 천장(Pity), 연패(Streak) 추가
# - Gambler Fallacy Trait: 도박 성향 추가
# ==========================================

NUM_MEDIA_TYPES = 6 

def create_agent_population(n_agents=10000):
    print(f"Creating {n_agents} agents with Deep Economy (v2.1)...")
    
    # 1. Static Traits
    traits_big5 = np.clip(np.random.normal(0.5, 0.15, (n_agents, 5)), 0.0, 1.0)
    loss_aversion = np.clip(np.random.normal(2.25, 0.5, (n_agents, 1)), 1.0, 5.0)
    traits_intel = np.clip(np.random.normal(50, 15, (n_agents, 1)), 0, 100)
    
    # [NEW] Gambler's Fallacy Trait (도박사의 오류 성향)
    # 높을수록 실패했을 때 "다음엔 무조건 된다"라고 믿음 (0.0 ~ 2.0)
    # 신경성(Big5[4])이 높을수록 도박 성향이 높게 설정
    neuroticism = traits_big5[:, 4].reshape(-1, 1)
    gambler_fallacy = np.clip(np.random.normal(1.0, 0.3, (n_agents, 1)) + (neuroticism * 0.5), 0.0, 2.0)
    
    # Attention Capacity
    base_cap = np.random.normal(100, 10, (n_agents, 1))
    conscientiousness = traits_big5[:, 1].reshape(-1, 1)
    attention_cap = np.clip(base_cap + (conscientiousness * 20), 50, 200).astype(int)

    # Life Pattern
    p_probs = [0.5, 0.3, 0.15, 0.05]
    life_pattern = np.random.choice([0, 1, 2, 3], size=(n_agents, 1), p=p_probs)
    
    # Wallet & Calibration
    wallet = np.random.lognormal(mean=10, sigma=1, size=(n_agents, 1)).astype(int)
    is_student = (life_pattern == 1).flatten()
    wallet[is_student] = (wallet[is_student] * 0.3).astype(int)
    is_free = (life_pattern == 2).flatten()
    wallet[is_free] = (wallet[is_free] * 0.5).astype(int)

    # 2. Dynamic States
    state_stress = np.zeros((n_agents, 1))
    state_fatigue = np.zeros((n_agents, 1))
    state_boredom = np.zeros((n_agents, 1))
    
    state_anxiety = np.random.uniform(0, 10, (n_agents, 1))
    state_anxiety[is_student] += 5.0
    state_dopamine = np.full((n_agents, 1), 50.0)
    
    # Context State
    state_current_media = np.full((n_agents, 1), -1, dtype=int)
    media_boredom = np.zeros((n_agents, NUM_MEDIA_TYPES))

    # [NEW] Gacha States
    gacha_pity_count = np.zeros((n_agents, 1), dtype=int) # 천장 스택
    recent_fail_streak = np.zeros((n_agents, 1), dtype=int) # 연속 실패

    # Interests
    interests = np.random.rand(n_agents, 50)
    mask = np.random.rand(n_agents, 50) > 0.3
    interests[mask] = 0.0
    
    population = {
        "ids": np.arange(n_agents),
        "life_pattern": life_pattern,
        "traits_big5": traits_big5,
        "traits_intel": traits_intel,
        "loss_aversion": loss_aversion,
        "gambler_fallacy": gambler_fallacy, # [NEW]
        "attention_cap": attention_cap,
        
        "state_stress": state_stress,
        "state_fatigue": state_fatigue,
        "state_boredom": state_boredom,
        "state_anxiety": state_anxiety,
        "state_dopamine": state_dopamine,
        "state_current_media": state_current_media,
        "media_boredom": media_boredom,
        
        "gacha_pity_count": gacha_pity_count,     # [NEW]
        "recent_fail_streak": recent_fail_streak, # [NEW]
        
        "wallet": wallet,
        "interests": interests
    }
    
    return population