import pandas as pd
import os

# ==========================================
# Initial Data Creator
# ==========================================
# 이 스크립트는 시뮬레이션에 필요한 초기 CSV 파일들을
# ./data 폴더에 생성합니다. (최초 1회 실행)
# ==========================================

def create_initial_csvs():
    # 폴더 생성
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created directory: ./data")

    # 1. Activity Table Data
    activities_data = [
        {"ID": "ACT_GM_PVP",   "Name": "PVP 랭킹전",    "Category": "GAME", "Intensity": 90, "Base_Reward": 50.0, "Cost": 0,    "Stress_Cost": 15.0, "Tags": "Competition|Skill"},
        {"ID": "ACT_GM_AUTO",  "Name": "자동사냥(방치)", "Category": "GAME", "Intensity": 15, "Base_Reward": 5.0,  "Cost": 0,    "Stress_Cost": 0.0,  "Tags": "Growth|RPG"},
        {"ID": "ACT_GM_GACHA", "Name": "확률형 아이템 뽑기", "Category": "GAME", "Intensity": 40, "Base_Reward": 80.0, "Cost": 3000, "Stress_Cost": -5.0, "Tags": "Gambling|Collection"},
        {"ID": "ACT_GM_AD",    "Name": "보상형 광고 시청", "Category": "GAME", "Intensity": 30, "Base_Reward": 2.0,  "Cost": 0,    "Stress_Cost": 5.0,  "Tags": "Free|Patience"},
        {"ID": "ACT_MD_SHORT", "Name": "숏폼(틱톡/릴스)",  "Category": "MEDIA", "Intensity": 60, "Base_Reward": 25.0, "Cost": 0,    "Stress_Cost": 5.0,  "Tags": "Humor|Trend"},
        {"ID": "ACT_CM_BOARD", "Name": "커뮤니티 눈팅",    "Category": "COMM",  "Intensity": 40, "Base_Reward": 10.0, "Cost": 0,    "Stress_Cost": 10.0, "Tags": "Social|Info"},
        {"ID": "ACT_LF_WORK",  "Name": "집중 업무/공부",   "Category": "WORK",  "Intensity": 85, "Base_Reward": 0.0,  "Cost": -200, "Stress_Cost": 20.0, "Tags": "Responsibility"},
        {"ID": "ACT_LF_REST",  "Name": "멍때리기/휴식",    "Category": "LIFE",  "Intensity": 5,  "Base_Reward": 1.0,  "Cost": 0,    "Stress_Cost": -20.0, "Tags": "Relax"},
    ]
    
    df_act = pd.DataFrame(activities_data)
    df_act.to_csv('data/activities.csv', index=False, encoding='utf-8-sig')
    print("-> Created: data/activities.csv")

    # 2. Time Slot Table Data
    slots = []
    for i in range(96):
        hour = (i * 15) // 60
        context = "NORMAL"
        stress_mod = 1.0
        ad_efficiency = 1.0
        
        if 0 <= hour < 7:     # 심야
            context = "SLEEP"
            stress_mod = 0.5
            ad_efficiency = 0.1
        elif 7 <= hour < 9:   # 출근
            context = "COMMUTE_AM"
            stress_mod = 1.5
            ad_efficiency = 1.5 
        elif 9 <= hour < 12:  # 오전 근무
            context = "WORK_AM"
            stress_mod = 1.2
            ad_efficiency = 0.5
        elif 12 <= hour < 13: # 점심
            context = "LUNCH"
            stress_mod = 0.8
            ad_efficiency = 1.2
        elif 13 <= hour < 18: # 오후 근무
            context = "WORK_PM"
            stress_mod = 1.3
            ad_efficiency = 0.5
        elif 18 <= hour < 20: # 퇴근
            context = "COMMUTE_PM"
            stress_mod = 1.4
            ad_efficiency = 1.4
        else:                 # 저녁 여가
            context = "RELAX"
            stress_mod = 0.9
            ad_efficiency = 1.1
            
        slots.append([i, hour, context, stress_mod, ad_efficiency])
    
    df_time = pd.DataFrame(slots, columns=["Time_Index", "Hour", "Context", "Stress_Mod", "Ad_Efficiency"])
    df_time.to_csv('data/time_slots.csv', index=False, encoding='utf-8-sig')
    print("-> Created: data/time_slots.csv")

if __name__ == "__main__":
    create_initial_csvs()