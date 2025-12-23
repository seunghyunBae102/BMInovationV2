import pandas as pd
import os

# ==========================================
# Initial Data Creator v1.5
# ==========================================
# [Update Log]
# - Media_Group: 매체 상위 분류 추가 (GAME, VIDEO, BOOK 등)
# - Fun_Reward / Growth_Reward: 보상 이원화 (재미 vs 성장)
# - Difficulty: 진입 장벽 (숙련도 요구치)
# ==========================================

def create_initial_csvs():
    # 폴더 생성
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created directory: ./data")

    # 1. Activity Table Data (Schema Updated)
    activities_data = [
        # --- GAME (도파민 위주, 난이도 다양) ---
        {
            "ID": "ACT_GM_PVP", "Name": "PVP 랭킹전", "Category": "GAME", 
            "Media_Group": "GAME", "Intensity": 90, "Fun_Reward": 60.0, "Growth_Reward": 5.0, "Difficulty": 70, # 이기기 어려움
            "Cost": 0, "Stress_Cost": 20.0, "Tags": "Competition|Skill"
        },
        {
            "ID": "ACT_GM_AUTO", "Name": "자동사냥(방치)", "Category": "GAME", 
            "Media_Group": "GAME", "Intensity": 15, "Fun_Reward": 10.0, "Growth_Reward": 2.0, "Difficulty": 5, # 매우 쉬움
            "Cost": 0, "Stress_Cost": 0.0, "Tags": "Growth|RPG"
        },
        {
            "ID": "ACT_GM_GACHA", "Name": "아이템 뽑기", "Category": "GAME", 
            "Media_Group": "GAME", "Intensity": 40, "Fun_Reward": 100.0, "Growth_Reward": 0.0, "Difficulty": 5, # 돈만 있으면 됨
            "Cost": 3000, "Stress_Cost": -5.0, "Tags": "Gambling|Collection"
        },
        
        # --- VIDEO (도파민 위주, 난이도 최하) ---
        {
            "ID": "ACT_MD_SHORT", "Name": "숏폼(틱톡/릴스)", "Category": "MEDIA", 
            "Media_Group": "VIDEO", "Intensity": 60, "Fun_Reward": 70.0, "Growth_Reward": -5.0, "Difficulty": 5, # 뇌 빼고 보기 가능
            "Cost": 0, "Stress_Cost": 5.0, "Tags": "Humor|Trend"
        },
        {
            "ID": "ACT_MD_NETFLIX", "Name": "넷플릭스 정주행", "Category": "MEDIA", 
            "Media_Group": "VIDEO", "Intensity": 40, "Fun_Reward": 40.0, "Growth_Reward": 2.0, "Difficulty": 10, 
            "Cost": 0, "Stress_Cost": -10.0, "Tags": "Story|Relax"
        },

        # --- BOOK/STUDY (성장 위주, 난이도 최상) ---
        {
            "ID": "ACT_BK_STUDY", "Name": "전공 공부/독서", "Category": "WORK", 
            "Media_Group": "BOOK", "Intensity": 85, "Fun_Reward": 5.0, "Growth_Reward": 80.0, "Difficulty": 75, # 지능/배경지식 필요
            "Cost": -10, "Stress_Cost": 25.0, "Tags": "Knowledge|Future"
        },
        
        # --- COMM (중간 성격) ---
        {
            "ID": "ACT_CM_BOARD", "Name": "커뮤니티 눈팅", "Category": "COMM", 
            "Media_Group": "COMM", "Intensity": 35, "Fun_Reward": 20.0, "Growth_Reward": 5.0, "Difficulty": 20, # 밈 이해도 필요
            "Cost": 0, "Stress_Cost": 5.0, "Tags": "Social|Info"
        },

        # --- LIFE ---
        {
            "ID": "ACT_LF_WORK", "Name": "집중 업무", "Category": "WORK", 
            "Media_Group": "WORK", "Intensity": 90, "Fun_Reward": 0.0, "Growth_Reward": 40.0, "Difficulty": 60,
            "Cost": -200, "Stress_Cost": 30.0, "Tags": "Responsibility"
        },
        {
            "ID": "ACT_LF_REST", "Name": "멍때리기/휴식", "Category": "LIFE", 
            "Media_Group": "LIFE", "Intensity": 5, "Fun_Reward": 5.0, "Growth_Reward": 5.0, "Difficulty": 0,
            "Cost": 0, "Stress_Cost": -20.0, "Tags": "Relax"
        },
    ]
    
    # 컬럼 순서 명시적 지정
    cols = ["ID", "Name", "Category", "Media_Group", "Intensity", "Fun_Reward", "Growth_Reward", "Difficulty", "Cost", "Stress_Cost", "Tags"]
    df_act = pd.DataFrame(activities_data, columns=cols)
    df_act.to_csv('data/activities.csv', index=False, encoding='utf-8-sig')
    print("-> Updated: data/activities.csv with Media Hierarchy schema")

    # 2. Time Slot Table (기존 유지)
    slots = []
    for i in range(96):
        hour = (i * 15) // 60
        context = "NORMAL"
        stress_mod = 1.0
        ad_efficiency = 1.0
        
        if 0 <= hour < 7:     context, stress_mod, ad_efficiency = "SLEEP", 0.5, 0.1
        elif 7 <= hour < 9:   context, stress_mod, ad_efficiency = "COMMUTE_AM", 1.5, 1.5 
        elif 9 <= hour < 12:  context, stress_mod, ad_efficiency = "WORK_AM", 1.2, 0.5
        elif 12 <= hour < 13: context, stress_mod, ad_efficiency = "LUNCH", 0.8, 1.2
        elif 13 <= hour < 18: context, stress_mod, ad_efficiency = "WORK_PM", 1.3, 0.5
        elif 18 <= hour < 20: context, stress_mod, ad_efficiency = "COMMUTE_PM", 1.4, 1.4
        else:                 context, stress_mod, ad_efficiency = "RELAX", 0.9, 1.1
            
        slots.append([i, hour, context, stress_mod, ad_efficiency])
    
    df_time = pd.DataFrame(slots, columns=["Time_Index", "Hour", "Context", "Stress_Mod", "Ad_Efficiency"])
    df_time.to_csv('data/time_slots.csv', index=False, encoding='utf-8-sig')
    print("-> Updated: data/time_slots.csv")

if __name__ == "__main__":
    create_initial_csvs()