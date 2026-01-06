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
            "Media_Group": "VIDEO", "Intensity": 20, "Fun_Reward": 95.0, "Growth_Reward": -5.0, "Difficulty": 5, # 뇌 빼고 보기 가능
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

    # 2. [NEW] Life Patterns Table
    # 패턴 ID: 0(직장인), 1(학생), 2(프리랜서), 3(야간조)
    patterns = []
    
    for i in range(96):
        hour = (i * 15) // 60
        
        # --- Pattern 0: 직장인 (Standard) ---
        if 0 <= hour < 7:     ctx, s_mod, ad = "SLEEP", 0.5, 0.1
        elif 7 <= hour < 9:   ctx, s_mod, ad = "COMMUTE", 1.5, 1.5 
        elif 9 <= hour < 18:  ctx, s_mod, ad = "WORK", 1.3, 0.5
        elif 18 <= hour < 20: ctx, s_mod, ad = "COMMUTE", 1.4, 1.4
        else:                 ctx, s_mod, ad = "RELAX", 0.9, 1.2
        patterns.append({"Pattern_ID": 0, "Time_Index": i, "Hour": hour, "Context": ctx, "Stress_Mod": s_mod, "Ad_Eff": ad})

        # --- Pattern 1: 학생 (Student) ---
        # 아침 일찍 등교, 수업 중 폰 압수(Ad효율 극악), 밤늦게까지 깨어있음
        if 0 <= hour < 7:     ctx, s_mod, ad = "SLEEP", 0.5, 0.1
        elif 7 <= hour < 8:   ctx, s_mod, ad = "COMMUTE", 1.2, 1.2
        elif 8 <= hour < 16:  ctx, s_mod, ad = "SCHOOL", 1.4, 0.2 # 수업중
        elif 16 <= hour < 22: ctx, s_mod, ad = "ACADEMY", 1.3, 0.8 # 학원/자습
        else:                 ctx, s_mod, ad = "GAME_TIME", 0.8, 1.5 # 새벽 몰래 게임
        patterns.append({"Pattern_ID": 1, "Time_Index": i, "Hour": hour, "Context": ctx, "Stress_Mod": s_mod, "Ad_Eff": ad})

        # --- Pattern 2: 프리랜서/백수 (Free) ---
        # 늦게 일어나고 낮에 놈
        if 0 <= hour < 11:    ctx, s_mod, ad = "SLEEP", 0.4, 0.0
        elif 11 <= hour < 18: ctx, s_mod, ad = "FREE_TIME", 0.8, 1.2 # 낮 게임
        elif 18 <= hour < 24: ctx, s_mod, ad = "FREE_TIME", 0.9, 1.2
        patterns.append({"Pattern_ID": 2, "Time_Index": i, "Hour": hour, "Context": ctx, "Stress_Mod": s_mod, "Ad_Eff": ad})
        
        # --- Pattern 3: 야간 근무자 (Night Shift) ---
        # 낮에 자고 밤에 일함
        if 9 <= hour < 17:    ctx, s_mod, ad = "SLEEP", 0.5, 0.1
        elif 17 <= hour < 20: ctx, s_mod, ad = "FREE_TIME", 0.8, 1.2
        elif 20 <= hour < 22: ctx, s_mod, ad = "COMMUTE", 1.2, 1.4
        elif 22 <= hour < 24: ctx, s_mod, ad = "WORK", 1.4, 0.4
        elif 0 <= hour < 6:   ctx, s_mod, ad = "WORK", 1.5, 0.4
        else:                 ctx, s_mod, ad = "COMMUTE", 1.4, 1.4
        patterns.append({"Pattern_ID": 3, "Time_Index": i, "Hour": hour, "Context": ctx, "Stress_Mod": s_mod, "Ad_Eff": ad})

    df_patterns = pd.DataFrame(patterns)
    df_patterns.to_csv('data/life_patterns.csv', index=False, encoding='utf-8-sig')
    print("-> Created: data/life_patterns.csv (4 Patterns)")

if __name__ == "__main__":
    create_initial_csvs()