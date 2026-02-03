#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习气象归一化处理工具
ML-Based Meteorological Normalization Tool

基于随机森林和XGBoost的气象归一化，用于分离排放和气象影响
Based on Random Forest and XGBoost for separating emission and meteorological effects
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MLMeteorologicalNormalization:
    """
    机器学习气象归一化处理类
    ML-Based Meteorological Normalization Class
    
    物理意义 (Physical Meaning):
    - C_norm (归一化浓度): 代表排放强度或污染源贡献 (Emission intensity or source contribution)
    - C_observed - C_norm: 代表气象因素对浓度波动的贡献绝对值 (Meteorological contribution)
    """
    
    def __init__(self, data_file, pollutant='pm2.5', model_type='rf', n_samples=1000):
        """
        初始化 / Initialize
        
        Parameters:
        -----------
        data_file : str
            数据文件路径 / Path to data file
        pollutant : str
            要处理的污染物 / Pollutant to process (pm10, pm2.5, SO2, NO2, CO, O3)
        model_type : str
            模型类型 / Model type ('rf' for Random Forest, 'xgb' for XGBoost)
        n_samples : int
            随机抽样次数 / Number of random samples for deweathering (default: 1000)
        """
        self.data_file = data_file
        self.pollutant = pollutant
        self.model_type = model_type
        self.n_samples = n_samples
        self.df = None
        self.model = None
        self.feature_importance = None
        
        # 气象变量和时间变量
        self.meteorological_vars = ['sp', 'temp', 'wd', 'ws', 'rh', 'rf', 'tcc']
        self.temporal_vars = ['year', 'month', 'day', 'hour', 'weekday', 'day_julian']
        self.pollutant_vars = ['pm10', 'pm2.5', 'SO2', 'NO2', 'CO', 'O3']
        
    def load_data(self):
        """加载数据 / Load data"""
        print(f"正在加载数据文件: {self.data_file}")
        print(f"Loading data file: {self.data_file}")
        self.df = pd.read_excel(self.data_file)
        
        # 转换日期列
        if 'data.1' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['data.1'])
        
        print(f"数据加载完成！共 {len(self.df)} 行，{len(self.df.columns)} 列")
        print(f"Data loaded! Total {len(self.df)} rows, {len(self.df.columns)} columns")
        return self
    
    def preprocess_data(self):
        """
        数据预处理 / Data preprocessing
        """
        print("\n正在进行数据预处理...")
        print("Preprocessing data...")
        
        # 将污染物列转换为数值类型
        for col in self.pollutant_vars:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 添加季节特征（北半球）
        # Add seasonal features (Northern Hemisphere)
        if 'month' in self.df.columns:
            self.df['season'] = self.df['month'].map({
                12: 0, 1: 0, 2: 0,  # 冬季 Winter
                3: 1, 4: 1, 5: 1,   # 春季 Spring
                6: 2, 7: 2, 8: 2,   # 夏季 Summer
                9: 3, 10: 3, 11: 3  # 秋季 Autumn
            })
            self.temporal_vars.append('season')
        
        # 添加小时的正弦余弦编码（处理循环特征）
        if 'hour' in self.df.columns:
            self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
            self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
            self.temporal_vars.extend(['hour_sin', 'hour_cos'])
        
        # 记录缺失值
        missing_counts = self.df[self.meteorological_vars + [self.pollutant]].isnull().sum()
        print(f"\n各列缺失值数量 / Missing values per column:")
        print(missing_counts[missing_counts > 0])
        
        # 删除目标污染物或气象变量缺失的行
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=self.meteorological_vars + [self.pollutant])
        removed_rows = initial_rows - len(self.df)
        
        if removed_rows > 0:
            print(f"\n删除了 {removed_rows} 行数据缺失的记录")
            print(f"Removed {removed_rows} rows with missing data")
        
        print(f"预处理后数据: {len(self.df)} 行")
        print(f"Data after preprocessing: {len(self.df)} rows")
        
        return self
    
    def build_model(self, test_size=0.2, random_state=42):
        """
        构建并训练机器学习模型 / Build and train ML model
        
        Parameters:
        -----------
        test_size : float
            测试集比例 / Test set ratio
        random_state : int
            随机种子 / Random seed
        """
        print(f"\n正在构建{self.model_type.upper()}模型...")
        print(f"Building {self.model_type.upper()} model...")
        
        # 准备特征和目标变量
        feature_cols = self.meteorological_vars + self.temporal_vars
        X = self.df[feature_cols].copy()
        y = self.df[self.pollutant].copy()
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集大小 Train set size: {len(X_train)}")
        print(f"测试集大小 Test set size: {len(X_test)}")
        
        # 构建模型
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
                verbose=0
            )
        elif self.model_type == 'xgb':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 训练模型
        print("\n训练模型中... Training model...")
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print("\n模型评估结果 / Model Evaluation:")
        print(f"训练集 R² (Train R²): {train_r2:.4f}")
        print(f"测试集 R² (Test R²): {test_r2:.4f}")
        print(f"训练集 RMSE (Train RMSE): {train_rmse:.4f}")
        print(f"测试集 RMSE (Test RMSE): {test_rmse:.4f}")
        print(f"训练集 MAE (Train MAE): {train_mae:.4f}")
        print(f"测试集 MAE (Test MAE): {test_mae:.4f}")
        
        # 获取特征重要性
        if self.model_type == 'rf':
            self.feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif self.model_type == 'xgb':
            self.feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print("\n特征重要性 Top 10 / Feature Importance Top 10:")
        print(self.feature_importance.head(10))
        
        return self
    
    def deweather(self):
        """
        执行气象归一化（解耦过程）/ Perform meteorological normalization (deweathering)
        
        对每个时间点，保持时间特征不变，随机抽取气象观测值进行多次预测并取平均
        For each time point, keep temporal features constant, randomly sample meteorological values
        """
        print(f"\n正在执行气象归一化处理（{self.n_samples}次随机抽样）...")
        print(f"Performing deweathering ({self.n_samples} random samples)...")
        
        feature_cols = self.meteorological_vars + self.temporal_vars
        
        # 存储归一化结果
        normalized_concentrations = []
        
        # 对每个时间点进行处理
        total_points = len(self.df)
        print_interval = max(1, total_points // 20)  # 打印20次进度
        
        for idx, row in self.df.iterrows():
            if (idx + 1) % print_interval == 0:
                progress = (idx + 1) / total_points * 100
                print(f"进度 Progress: {progress:.1f}% ({idx + 1}/{total_points})")
            
            # 保持时间特征不变
            temporal_features = row[self.temporal_vars].values
            
            # 从整个数据集中随机抽取气象数据
            random_indices = np.random.choice(len(self.df), size=self.n_samples, replace=True)
            random_meteo = self.df.iloc[random_indices][self.meteorological_vars].values
            
            # 构建预测特征：时间特征不变 + 随机气象特征
            # 时间特征需要重复n_samples次
            temporal_repeated = np.tile(temporal_features, (self.n_samples, 1))
            
            # 合并特征
            X_deweather = np.hstack([random_meteo, temporal_repeated])
            
            # 预测
            predictions = self.model.predict(X_deweather)
            
            # 计算平均值作为归一化浓度（排放强度）
            normalized_conc = predictions.mean()
            normalized_concentrations.append(normalized_conc)
        
        # 添加归一化结果到数据框
        self.df[f'{self.pollutant}_normalized'] = normalized_concentrations
        
        # 计算气象贡献
        self.df[f'{self.pollutant}_meteo_contribution'] = (
            self.df[self.pollutant] - self.df[f'{self.pollutant}_normalized']
        )
        
        print("\n气象归一化完成！/ Deweathering completed!")
        print(f"\n归一化结果统计 / Normalized Results Statistics:")
        print(f"原始浓度均值 (Original mean): {self.df[self.pollutant].mean():.2f}")
        print(f"归一化浓度均值 (Normalized mean): {self.df[f'{self.pollutant}_normalized'].mean():.2f}")
        print(f"气象贡献均值 (Meteorological contribution mean): "
              f"{self.df[f'{self.pollutant}_meteo_contribution'].mean():.2f}")
        print(f"气象贡献标准差 (Meteorological contribution std): "
              f"{self.df[f'{self.pollutant}_meteo_contribution'].std():.2f}")
        
        return self
    
    def plot_comparison(self, start_date=None, end_date=None, save_path=None):
        """
        绘制对比图 / Plot comparison
        
        Parameters:
        -----------
        start_date : str, optional
            起始日期 / Start date (format: 'YYYY-MM-DD')
        end_date : str, optional
            结束日期 / End date (format: 'YYYY-MM-DD')
        save_path : str, optional
            保存路径 / Save path
        """
        print("\n正在生成对比图表...")
        print("Generating comparison plots...")
        
        # 数据筛选
        plot_df = self.df.copy()
        if start_date:
            plot_df = plot_df[plot_df['datetime'] >= start_date]
        if end_date:
            plot_df = plot_df[plot_df['datetime'] <= end_date]
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'{self.pollutant.upper()} 气象归一化分析 / Meteorological Normalization Analysis',
                     fontsize=16, fontweight='bold')
        
        # 图1: 原始观测值 vs 归一化浓度
        ax1 = axes[0]
        ax1.plot(plot_df['datetime'], plot_df[self.pollutant], 
                label='原始观测值 / Original Observations', color='blue', alpha=0.6, linewidth=1)
        ax1.plot(plot_df['datetime'], plot_df[f'{self.pollutant}_normalized'],
                label='归一化浓度（排放强度）/ Normalized (Emission Intensity)', 
                color='red', alpha=0.8, linewidth=1.5)
        ax1.set_ylabel(f'{self.pollutant.upper()} 浓度 / Concentration', fontsize=12)
        ax1.set_title('原始观测值 vs 归一化浓度 / Original vs Normalized', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 图2: 气象贡献值
        ax2 = axes[1]
        colors = ['green' if x >= 0 else 'orange' for x in plot_df[f'{self.pollutant}_meteo_contribution']]
        ax2.fill_between(plot_df['datetime'], 0, plot_df[f'{self.pollutant}_meteo_contribution'],
                        color='green', alpha=0.3, label='气象促进 / Meteorological Enhancement')
        ax2.fill_between(plot_df['datetime'], 0, plot_df[f'{self.pollutant}_meteo_contribution'],
                        where=plot_df[f'{self.pollutant}_meteo_contribution'] < 0,
                        color='orange', alpha=0.3, label='气象抑制 / Meteorological Suppression')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel('气象贡献值 / Meteorological Contribution', fontsize=12)
        ax2.set_title('气象因素对浓度的影响 / Meteorological Impact on Concentration', 
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 图3: 三者叠加对比
        ax3 = axes[2]
        ax3.plot(plot_df['datetime'], plot_df[self.pollutant], 
                label='原始观测值 / Original', color='blue', alpha=0.5, linewidth=1)
        ax3.plot(plot_df['datetime'], plot_df[f'{self.pollutant}_normalized'],
                label='归一化浓度 / Normalized', color='red', alpha=0.8, linewidth=2)
        ax3.plot(plot_df['datetime'], plot_df[f'{self.pollutant}_meteo_contribution'],
                label='气象贡献 / Meteorological Contribution', color='green', alpha=0.7, linewidth=1)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax3.set_xlabel('日期 / Date', fontsize=12)
        ax3.set_ylabel('浓度 / Concentration', fontsize=12)
        ax3.set_title('综合对比 / Comprehensive Comparison', fontsize=13, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
            print(f"Plot saved to: {save_path}")
        else:
            plt.savefig(f'{self.pollutant}_comparison.png', dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {self.pollutant}_comparison.png")
            print(f"Plot saved to: {self.pollutant}_comparison.png")
        
        plt.close()
        
        return self
    
    def plot_feature_importance(self, top_n=15, save_path=None):
        """
        绘制特征重要性图 / Plot feature importance
        
        Parameters:
        -----------
        top_n : int
            显示前N个特征 / Show top N features
        save_path : str, optional
            保存路径 / Save path
        """
        if self.feature_importance is None:
            print("请先训练模型 / Please train the model first")
            return
        
        print("\n正在生成特征重要性图...")
        print("Generating feature importance plot...")
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('重要性 / Importance', fontsize=12)
        plt.ylabel('特征 / Feature', fontsize=12)
        plt.title(f'{self.pollutant.upper()} 预测模型特征重要性 / Feature Importance',
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.savefig(f'{self.pollutant}_feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {self.pollutant}_feature_importance.png")
        
        plt.close()
        
        return self
    
    def save_results(self, output_file):
        """
        保存结果 / Save results
        
        Parameters:
        -----------
        output_file : str
            输出文件路径 / Output file path
        """
        print(f"\n正在保存结果到: {output_file}")
        print(f"Saving results to: {output_file}")
        
        # 选择要保存的列
        save_columns = ['datetime', 'year', 'month', 'day', 'hour', 
                       self.pollutant, 
                       f'{self.pollutant}_normalized',
                       f'{self.pollutant}_meteo_contribution'] + self.meteorological_vars
        
        self.df[save_columns].to_excel(output_file, index=False)
        print("保存完成！/ Save completed!")
        
        return self
    
    def get_summary_statistics(self):
        """
        获取汇总统计信息 / Get summary statistics
        
        Returns:
        --------
        dict : 统计信息字典 / Dictionary of statistics
        """
        stats = {
            'pollutant': self.pollutant,
            'model_type': self.model_type,
            'n_samples': self.n_samples,
            'original_mean': self.df[self.pollutant].mean(),
            'original_std': self.df[self.pollutant].std(),
            'normalized_mean': self.df[f'{self.pollutant}_normalized'].mean(),
            'normalized_std': self.df[f'{self.pollutant}_normalized'].std(),
            'meteo_contribution_mean': self.df[f'{self.pollutant}_meteo_contribution'].mean(),
            'meteo_contribution_std': self.df[f'{self.pollutant}_meteo_contribution'].std(),
            'correlation_obs_norm': self.df[[self.pollutant, f'{self.pollutant}_normalized']].corr().iloc[0, 1]
        }
        
        return stats
    
    def print_summary(self):
        """打印汇总信息 / Print summary"""
        stats = self.get_summary_statistics()
        
        print("\n" + "="*80)
        print("气象归一化分析总结 / Meteorological Normalization Summary")
        print("="*80)
        print(f"污染物 / Pollutant: {stats['pollutant'].upper()}")
        print(f"模型类型 / Model Type: {stats['model_type'].upper()}")
        print(f"随机抽样次数 / Random Samples: {stats['n_samples']}")
        print("-"*80)
        print("原始浓度统计 / Original Concentration Statistics:")
        print(f"  均值 Mean: {stats['original_mean']:.2f}")
        print(f"  标准差 Std: {stats['original_std']:.2f}")
        print("-"*80)
        print("归一化浓度统计（排放强度）/ Normalized Concentration (Emission Intensity):")
        print(f"  均值 Mean: {stats['normalized_mean']:.2f}")
        print(f"  标准差 Std: {stats['normalized_std']:.2f}")
        print("-"*80)
        print("气象贡献统计 / Meteorological Contribution Statistics:")
        print(f"  均值 Mean: {stats['meteo_contribution_mean']:.2f}")
        print(f"  标准差 Std: {stats['meteo_contribution_std']:.2f}")
        print("-"*80)
        print(f"观测值与归一化值相关系数 / Correlation (Obs vs Norm): {stats['correlation_obs_norm']:.4f}")
        print("="*80)


def main():
    """主函数 / Main function"""
    print("="*80)
    print("机器学习气象归一化处理")
    print("ML-Based Meteorological Normalization")
    print("="*80)
    
    # 配置参数
    data_file = '石家庄市污染物数据及气象数据.xlsx'
    pollutant = 'pm2.5'  # 可选: pm10, pm2.5, SO2, NO2, CO, O3
    model_type = 'rf'    # 可选: 'rf' (Random Forest) 或 'xgb' (XGBoost)
    n_samples = 1000     # 随机抽样次数
    
    # 创建归一化对象
    normalizer = MLMeteorologicalNormalization(
        data_file=data_file,
        pollutant=pollutant,
        model_type=model_type,
        n_samples=n_samples
    )
    
    # 执行完整流程
    normalizer.load_data()
    normalizer.preprocess_data()
    normalizer.build_model()
    normalizer.deweather()
    
    # 生成图表
    normalizer.plot_comparison(start_date='2025-01-01', end_date='2025-01-31')
    normalizer.plot_feature_importance()
    
    # 保存结果
    normalizer.save_results(f'{pollutant}_ml_normalized_results.xlsx')
    
    # 打印总结
    normalizer.print_summary()
    
    print("\n" + "="*80)
    print("处理完成！/ Processing completed!")
    print("="*80)


if __name__ == '__main__':
    main()
