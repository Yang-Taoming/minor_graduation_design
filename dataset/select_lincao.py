import re
import pandas as pd

def select_lincao(file_path,lincao_word):
    df = pd.read_csv(file_path,sep='\t',names=['news_id','category','subcategory','title','abstract','url','title_entities','abstract_entities'],quoting=3)
    print('original news length:',file_path,len(df))
    keywords = [word.strip() for word in lincao_word if str(word).strip()]
    if not keywords:
        df_select = df.iloc[0:0]
        print('selected news length:',file_path,len(df_select))
        return df_select
    # 使用单词边界匹配，避免子串误匹配
    pattern =r"\b(?:%s)\b" % "|".join(re.escape(word) for word in keywords)
    mask = df['abstract'].fillna('').str.contains(pattern, case=False, regex=True)
    df_select = df[mask]
    df_select_news_id = df_select['news_id'].astype(str).unique().tolist()
    print('selected news length:',file_path,len(df_select_news_id))
    return df_select_news_id

def select_behaviors(file_path,df_select_news_id,target_user_count=None):
    clicks = 0
    history_clicks = 0
    impressions = 0
    final_news_set = []  # 最终包含的新闻集
    df = pd.read_csv(file_path,sep='\t',names=['impression_id','user_id','time','history','impressions'],quoting=3)
    print('original behaviors length:',file_path,len(df))
    # 将 news_id 转换为集合，提高查找效率
    news_id_set = set(map(str, df_select_news_id))
    histories = df['history'].fillna('').astype(str).str.split()
    history_exploded = histories.explode()
    mask = history_exploded.isin(news_id_set).groupby(level=0).any()
    mask = mask.reindex(df.index, fill_value=False)
    user_select = df[mask]
    if target_user_count is not None:
        user_ids = user_select['user_id'].dropna().unique()
        if len(user_ids) > target_user_count:
            sampled_users = pd.Series(user_ids).sample(n=target_user_count, random_state=42)
            user_select = user_select[user_select['user_id'].isin(sampled_users)]

    selected_histories = histories.loc[user_select.index]
    history_clicks = selected_histories.str.len().sum()

    impressions_list = user_select['impressions'].fillna('').astype(str).str.split()
    impressions = len(user_select)

    impressions_exploded = impressions_list.explode()
    further_clicks = impressions_exploded.str.endswith('-1').groupby(level=0).sum()
    clicks = further_clicks.sum()

    history_ids = selected_histories.explode().dropna()
    impression_ids = impressions_exploded.dropna().str.split('-', n=1).str[0]
    final_news_set = set(history_ids)
    final_news_set.update(impression_ids)
    final_news_set = list(final_news_set)
    print('selected behaviors length:',file_path,len(user_select),clicks,history_clicks,impressions)
    print('final news set length:',len(final_news_set))
    return user_select,clicks,history_clicks,impressions,final_news_set

def select_final_news(file_path,final_news_set):
    df = pd.read_csv(file_path,sep='\t',names=['news_id','category','subcategory','title','abstract','url','title_entities','abstract_entities'],quoting=3)
    df_select = df[df['news_id'].isin(final_news_set)]
    return df_select

if __name__ == '__main__':
    front_file_path = r'dataset\MIND'
    mid_file_paths =['large_dev','large_train','small_dev','small_train']
    lincao_word = [
    # --- 林业基础词汇 ---
    "Forest",       # 森林
    "Forestry",     # 林业
    "Tree",         # 树木
    "Wood",         # 木材
    "Timber",       # 木材/林木
    "Stand",        # 林分
    "Canopy",       # 林冠
    "Understory",   # 林下层
    "Leaf",         # 叶片
    "Root",         # 根系
    "Soil",         # 土壤
    "Seed",         # 种子
    "Seedling",     # 幼苗
    "Nursery",      # 苗圃
    "Plantation",   # 人工林

    # --- 草地基础词汇 ---
    "Grassland",    # 草地
    "Pasture",      # 牧草地
    "Rangeland",    # 牧场
    "Meadow",       # 草甸
    "Shrub",        # 灌木
    "Shrubland",    # 灌丛地
    "Herb",         # 草本

    # --- 常见生态与管理词汇 ---
    "Ecosystem",    # 生态系统
    "Biodiversity", # 生物多样性
    "Habitat",      # 栖息地
    "Vegetation",   # 植被
    "Biomass",      # 生物量
    "Carbon",       # 碳
    "Water",        # 水
    "Erosion",      # 侵蚀
    "Restoration",  # 修复
    "Conservation", # 保护
    "Management",   # 管理
    "Afforestation",# 造林
    "Reforestation",# 复绿/再造林
    "Grazing",      # 放牧
    ]
    static_data = pd.DataFrame(
        columns = ['small_train','small_dev','large_train','large_dev'],
        index = ['users','news','clicks','history_clicks','impressions','average title len','average abs len']
    )
    target_user_counts = {
        'small_train': 8000,
        'small_dev': 2000,
        'large_train': 40000,
        'large_dev': 10000,
    }
    for mid_file_path in mid_file_paths:
        file_path = front_file_path+mid_file_path+r'\news.tsv'
        df_select_news_id = select_lincao(file_path,lincao_word)       
        file_path = file_path.replace(r'news.tsv',r'behaviors.tsv')
        user_select,clicks,history_clicks,impressions,final_news_set = select_behaviors(
            file_path,
            df_select_news_id,
            target_user_counts.get(mid_file_path)
        )
        file_path = file_path.replace(r'behaviors.tsv',r'news.tsv')
        final_news = select_final_news(file_path,final_news_set)
        final_news.to_csv(file_path.replace('.tsv','_lincao.tsv'),sep='\t',index=False)
        static_data.loc['clicks',mid_file_path] = clicks
        static_data.loc['history_clicks',mid_file_path] = history_clicks
        static_data.loc['impressions',mid_file_path] = impressions
        static_data.loc['users',mid_file_path] = len(user_select.user_id.unique())
        static_data.loc['news',mid_file_path] = len(final_news_set)
        static_data.loc['average title len',mid_file_path] = final_news['title'].fillna('').str.split().str.len().mean()
        static_data.loc['average abs len',mid_file_path] = final_news['abstract'].fillna('').str.split().str.len().mean()
        user_select.to_csv(file_path.replace('.tsv','_lincao.tsv'),sep='\t',index=False)
    static_data.to_csv('static_data_1.csv',index=True)
        