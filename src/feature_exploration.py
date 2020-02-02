import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/raw/intern_data.csv', index_col=0)
Y_LABEL = 'y'
num_cols = ['a', 'b', 'd', 'e', 'f', 'g']
ctg_cols = ['c', 'h']

df.isna().sum()

def plot_features_with_y(df_input):
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    sns.set()
    for i in range(len(num_cols)):
        col_name = num_cols[i]
        ax = plt.subplot(2, 3, i + 1)
        #ax.set_title(col_name)
        sns.scatterplot(x=col_name, y=Y_LABEL, data=df_input)
    plt.show()
plot_features_with_y(df)

df_white = df[df['h'] == 'white']
df_black = df[df['h'] == 'black']
plot_features_with_y(df_black)

def plot_feature_dist(df_input):
    fig, axs = plt.subplots(3, 3, figsize=(16, 8))
    for i in range(len(num_cols)):
        col_name = num_cols[i]
        ax = plt.subplot(3, 3, i + 1)
        #ax.set_title(col_name)
        sns.distplot(df_input[col_name], bins=30, hist=True, label=col_name)
    ax = plt.subplot(3, 3, 7)
    sns.distplot(df_input[Y_LABEL], bins=30, hist=True, label=Y_LABEL)
    axs[2, 1].axis('off')
    axs[2, 2].axis('off')
    plt.show()
plot_feature_dist(df)

plot_feature_dist(df_white)
plot_feature_dist(df_black)

df_dummy = pd.get_dummies(df, columns=ctg_cols)

df_dummy['ehc'] = df_dummy['e'] * df_dummy['h_white'] * df_dummy['c_blue'].map({0: 1, 1: 0})
df_dummy['e(h+c)'] = df_dummy['e'] * (df_dummy['h_white'] + df_dummy['c_blue'].map({0: 1, 1: 0}))
df_dummy['f+g'] = df_dummy['f'] + df_dummy['g']
df_dummy['(f+g)*h*c'] = (df_dummy['f'] + df_dummy['g']) * df_dummy['h_white'] * df_dummy['c_blue'].map({0: 1, 1: 0})
df_dummy['(f+g)*(h+c)'] = (df_dummy['f'] + df_dummy['g']) * (df_dummy['h_white'] + df_dummy['c_blue'].map({0: 1, 1: 0}))

df_dummy['ehc(f+g)'] = df_dummy['ehc'] * df_dummy['f+g']
df_dummy['e(h+c)(f+g)'] = df_dummy['e(h+c)'] * df_dummy['f+g']

df_dummy['ehc+f+g)'] = df_dummy['ehc'] + df_dummy['f+g']
df_dummy['e(h+c)+f+g)'] = df_dummy['e(h+c)'] + df_dummy['f+g']

df_dummy['eh(f+g)*(h+c)'] = df_dummy['ehc'] * df_dummy['(f+g)*(h+c)']
df_dummy['eh+(f+g)*(h+c)'] = df_dummy['ehc'] + df_dummy['(f+g)*(h+c)']

df_dummy['e(h+c)(f+g)*(h+c)'] = df_dummy['e(h+c)'] * df_dummy['(f+g)*(h+c)']
df_dummy['e(h+c)+(f+g)*(h+c)'] = df_dummy['e(h+c)'] + df_dummy['(f+g)*(h+c)']

fig, axs = plt.subplots(2, 3, figsize=(16, 8))
for i in range(len(num_cols)):
    col_name = num_cols[i]
    ax = plt.subplot(2, 3, i + 1)
    ax.set_title(col_name)
    bp = ax.boxplot(df[col_name], 0, '', showfliers=True)

colormap = plt.cm.RdBu
f, axs = plt.subplots(1, 1, figsize=(8, 8))
#ax = plt.subplot(3, 3, idx + 1)
sns.heatmap(df_dummy.corr(method='pearson', min_periods=1).round(decimals=2), linewidths=0.1, vmax=1.0, square=True,
            cmap=colormap, linecolor='white', annot=True)
#plt.xticks(rotation=30, fontsize=7)
plt.show()

colormap = plt.cm.RdBu
f, axs = plt.subplots(1, 1, figsize=(13, 13))
#ax = plt.subplot(3, 3, idx + 1)
sns.heatmap(df_dummy.corr(method='pearson', min_periods=1).round(decimals=2), linewidths=0.1, vmax=1.0, square=True,
            cmap=colormap, linecolor='white', annot=True)
#plt.xticks(rotation=30, fontsize=7)
plt.show()

pca = PCA(n_components=8)
pca_result = pca.fit_transform(df_dummy)
print(pca.explained_variance_ratio_)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Explained variance ratio")
plt.title("Explained Variance")
plt.xticks(np.arange(0, 3, 1))
plt.show()

