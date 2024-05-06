# import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from typing import Optional
# from io import BytesIO
# from gradio import File, Image
import pandas as pd
import seaborn as sns

def get_production_data() -> pd.DataFrame:
    df = pd.read_csv('.\Human_SingleCell_TrackingTable_20240410.csv', encoding='gb18030')
    data = df[['Cell ID', '拍摄日期', '拍摄人员', '脑区']].copy()

    return data

def get_manual_data() -> pd.DataFrame:
    df = pd.read_csv('.\Manual_2377_global_features.csv')
    
    return df

def get_auto_data() -> pd.DataFrame:
    df = pd.read_csv('.\Auto_10k_global_features.csv')
    
    return df


def plot_production_trend(data: pd.DataFrame, ax: Optional[plt.Axes] = None, title_name: str ='') -> plt.Axes:
    '''
    This function plots production trend.

    Args:
    -data: pandas DataFrame, 要绘制的库存数据。
    -ax: matplotlib Axes object, the axes to plot the data on 用于绘制数据的轴
    -title_name: 图表标题

    Returns:
    - matplotlib Axes object, the axes containing the plot
    '''

    print("Now we're in the 'plot_production_trend' function! ")

    # 数据处理
    for cell in data.index:
        date = data.loc[cell, '拍摄日期']
        year = date.split('-')[0]
        month = date.split('-')[1]
        day = date.split('-')[2]
        data.loc[cell, 'year-month'] = year + '-' + month
        data.loc[cell, 'year'] = int(year)
        data.loc[cell, 'month'] = int(month)
        data.loc[cell, 'day'] = int(day)
    
    print(cell)

    product_ana = data['year-month'].value_counts().sort_index()
    product_total_ana = np.zeros(product_ana.shape[0])
    for i, p in enumerate(product_ana):
        if i == 0:
            product_total_ana[i] = p
        else:
            product_total_ana[i] += (product_total_ana[i-1] + p)

    # 绘图
    if ax is None:
        _, ax = plt.subplots()

    x_ticks = range(len(product_ana))
    ax.plot(x_ticks, product_total_ana, '.', label='Total', color='red', linewidth=2, linestyle='-')
    ax.bar(x_ticks, product_ana, label='Monthly', color='#95d0fc')

    for i, count in enumerate(product_ana):
        ax.text(i, count + 300, '%d' % int(count), fontsize=15, ha='center')

    ax.set_xticks(x_ticks)  # 设置x轴刻度位置
    ax.set_xticklabels(product_ana.keys().tolist(), rotation=90, fontsize=15)  # 设置x轴刻度标签
    ax.yaxis.set_major_locator(MaxNLocator(12))
    ax.xaxis.set_minor_locator(MaxNLocator(len(product_total_ana)))
    ax.grid(False)  # 删除网格线

    ax.tick_params(axis='y', labelsize=15)
    ax.legend(fontsize=15)
    ax.set_xlabel('Month', fontsize=20)
    ax.set_ylabel('Number of Cells', fontsize=20)
    ax.set_title(title_name, fontsize=30, fontweight='bold')

    # plt.tight_layout()  
    fig = ax.figure  # 获取与ax关联的figure对象
    # fig = plt.gcf()
    fig.set_size_inches(18, 12)
    # plt.show()
    plt.savefig('StateofHumanBrainCells.png')

    return ax

    # # 绘图
    # # fig = plt.figure(figsize=(16, 9), dpi=120)
    # if ax is None:  # 如果没有提供Axes对象，则创建一个
    #     ax = plt.subplot(111)
    
    # ax.plot(product_ana.keys().tolist(), product_total_ana, '.', label='Total', color='red', linewidth=2, linestyle='-')
    # ax.bar(product_ana.keys(), product_ana, label='Monthly', color='#95d0fc')

    # for i, b in enumerate(product_ana.keys()):
    #     bb = plt.bar(i, product_ana[i], color='#95d0fc')
    #     for rect in bb:
    #         w = rect.get_height()
    #         if w > 0:
    #             ax.text(rect.get_x() + rect.get_width() / 5, w + 300, '%d' % int(w), fontsize=15)

    # ax.set_xticklabels(product_ana.keys().tolist(), rotation=90, fontsize=15)
    # ax.yaxis.set_major_locator(MaxNLocator(12))
    # ax.xaxis.set_minor_locator(MaxNLocator(len(product_total_ana)))
    # ax.grid(False)  # 删除网格线

    # plt.yticks(fontsize=15)
    # plt.legend(fontsize=15)
    # plt.xlabel('Month', fontsize=20)
    # plt.ylabel('Number of Cells', fontsize=20)
    # plt.title(title_name, fontsize=30, fontweight='bold')
    
    # plt.tight_layout()  
    # fig = ax.gcf()    # 获取与ax关联的figure对象
    # fig.set_size_inches(18, 12)

    # 保存图形
    # fig.savefig('C:\\Users\\kaixiang\\Desktop\\Fig\\StateofHumanBrainCells.png', dpi=120, format='png')

    # # 保存图形到BytesIO对象并返回
    # buf = BytesIO()
    # plt.savefig(buf, format="png")
    # buf.seek(0)

    # return buf

def plot_feature_distribution(data: pd.DataFrame, title_name: str ='') -> plt.Axes:
    '''
    This function plots features distribution.

    Args:
    -data: pandas DataFrame, 要绘制的库存数据。
    -title_name: 图表标题

    Returns:
    - matplotlib Axes object, the axes containing the plot
    '''

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 定义需要绘制的特征
    selected_features = ['Bifurcations', 'Length', 'MaxPathDistance', 'AverageContraction']

    # 创建画布，设置分辨率
    plt.figure(figsize=(18, 12))

    # 为选定的特征绘制直方图和KDE
    for i, feature in enumerate(selected_features):
        ax = plt.subplot(2, 2, i+1)  # 按2行2列排列
        # bins = 250 if feature == 'Volume' else 20
        sns.histplot(data=data, x=feature, kde=True, bins=20, color='#3287d6', edgecolor='black')
        ax.set_title(feature, fontsize=18)  # 子图标题
        ax.set_xlabel('Value')  # x轴标签
        ax.set_ylabel('Count')  # y轴标签

    # plt.suptitle(f'Auto-10k Neuron Features Distribution', fontsize=35, fontweight='bold')
    # ax.set_title(title_name, fontsize=30, fontweight='bold')
    plt.suptitle(title_name, fontsize=30, fontweight='bold')
    plt.tight_layout()
    plt.savefig("FeaturesDistribution.png")

    return ax

def plot_version_comparison(data1: pd.DataFrame, data2: pd.DataFrame, title_name: str ='') -> plt.Axes:
    '''
    Comparison between manual and auto versions 

    Args:
    -data1, data2: pandas DataFrame, 要绘制的库存数据。
    -title_name: 图表标题

    Returns:
    - matplotlib Axes object, the axes containing the plot
    '''
   # 加载数据
    # manual_df = data1
    # auto_df = data2

    # print(type(data1))
    # print(data1['Name']).head
    # print(type(data2))
    # print(data2['Name']).head

    # # print(type(manual_df))
    # # print(manual_df['Name']).head

    # # 仅保留 '.' 之前的内容
    # data1['Name'] = data1['Name'].apply(lambda x: x.split('.')[0])
    # data2.loc['Name'] = data2.loc['Name'].apply(lambda x: x.split('.')[0])

    # 找到编号相同的数据
    merged_df = pd.merge(data1, data2, on='Name', suffixes=('_Manual', '_Auto'))

    # 确保数据正确合并
    if merged_df.empty:
        raise ValueError("合并后的 DataFrame 为空，请检查 'Name' 列的匹配项。")

    # 选择特定的特征进行分析
    selected_features = ['Bifurcations', 'Length', 'AverageContraction', 'MaxPathDistance', 'MaxBranchOrder', 'AverageBifurcationAngleLocal']
    selected_columns = [f"{feature}_Manual" for feature in selected_features] + [f"{feature}_Auto" for feature in selected_features]

    # 只保留选择的特征列和Name列
    merged_df = merged_df[['Name'] + selected_columns]

    # 准备长格式的DataFrame以便于绘图
    melted_df = pd.melt(merged_df, id_vars=['Name'], var_name='Features', value_name='Value')

    # 添加一个新的列以区分手动重建和自动重建
    melted_df['Reconstruction Method'] = melted_df['Features'].apply(lambda x: 'Manual' if '_Manual' in x else 'Auto')

    # 添加一个特征名列，去除后缀，并确保没有重复的处理
    melted_df['Feature'] = melted_df['Features'].str.extract(r'(.+?)_(Manual|Auto)')[0]

    # 设置绘图风格
    sns.set(style="whitegrid")
    
    # plt.figure(figsize=(16, 9), dpi=120)

    # 设置画布大小
    fig, axs = plt.subplots(3, 2, figsize=(18, 12))  # 3x2布局

    # 绘制每个特征的小提琴图
    for ax, feature in zip(axs.flat, selected_features):
        feature_data = melted_df[melted_df['Feature'] == feature]
        sns.violinplot(feature_data, x='Reconstruction Method', y='Value', ax=ax)
        ax.set_title(feature, fontsize=15)
        ax.set_xlabel('', fontsize=14)
        ax.set_ylabel('Value', fontsize=12)

    # plt.suptitle('Comparison Between Manual and Automatic Versions', fontsize=30, fontweight='bold')
    plt.suptitle(title_name, fontsize=30, fontweight='bold')
    plt.tight_layout() 

    plt.savefig("VersionsComparison")

    return fig


# # 创建Gradio接口
# iface = gr.Interface(fn = process_and_visualize,
#                      inputs = File(label="上传CSV文件"),
#                      outputs = Image(label="人脑细胞状态"),
#                      title = "人脑细胞生产状态可视化",
#                      description = "上传Human_SingleCell_TrackingTable.csv文件, 查看人脑细胞生产的月度和累计情况。")

# # 启动应用
# iface.launch()

# if __name__ == "__main__":
#     ax = None

#     if ax is not None:
#         plt.grid()
#         plt.show()


