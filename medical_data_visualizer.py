import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = ((df['weight']/(df['height']*0.01)**2) > 25).astype(int)

# 3
df.loc[:,['cholesterol','gluc']] = df.loc[:, ['cholesterol', 'gluc']].map(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = df.melt(value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], id_vars=['cardio'])


    # 6
    df_cat = df_cat.groupby('cardio').value_counts()
    

    # 7
    df_cat = df_cat.reset_index(name = 'total')

    # 8
    fig = sns.catplot(df_cat, 
                x='variable', y='total', hue='value', 
                col='cardio', kind='bar').figure


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu((np.ones(corr.shape))).astype(bool)



    # 14
    fig, ax = plt.subplots(figsize = (12, 10))

    # 15
    sns.heatmap(corr, mask = mask, annot = True, fmt = '.1f', ax = ax)


    # 16
    fig.savefig('heatmap.png')
    return fig
