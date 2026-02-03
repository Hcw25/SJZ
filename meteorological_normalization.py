#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气象归一化处理工具
Meteorological Normalization Processing Tool

此脚本用于对石家庄市污染物数据进行气象归一化处理
This script is used for meteorological normalization of Shijiazhuang pollutant data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MeteorologicalNormalization:
    """气象归一化处理类 / Meteorological Normalization Class"""
    
    def __init__(self, data_file):
        """
        初始化 / Initialize
        
        Parameters:
        -----------
        data_file : str
            数据文件路径 / Path to data file
        """
        self.data_file = data_file
        self.df = None
        self.scaler = StandardScaler()
        self.meteorological_vars = ['sp', 'temp', 'wd', 'ws', 'rh', 'rf', 'tcc']
        self.pollutant_vars = ['pm10', 'pm2.5', 'SO2', 'NO2', 'CO', 'O3']
        
    def load_data(self):
        """加载数据 / Load data"""
        print(f"正在加载数据文件: {self.data_file}")
        print(f"Loading data file: {self.data_file}")
        self.df = pd.read_excel(self.data_file)
        print(f"数据加载完成！共 {len(self.df)} 行，{len(self.df.columns)} 列")
        print(f"Data loaded! Total {len(self.df)} rows, {len(self.df.columns)} columns")
        return self
    
    def preprocess_data(self):
        """
        数据预处理 / Data preprocessing
        处理污染物数据中的缺失值和异常值
        """
        print("\n正在进行数据预处理...")
        print("Preprocessing data...")
        
        # 将污染物列转换为数值类型
        # Convert pollutant columns to numeric type
        for col in self.pollutant_vars:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 记录缺失值数量
        # Record missing values
        missing_counts = self.df[self.meteorological_vars + self.pollutant_vars].isnull().sum()
        print(f"\n各列缺失值数量 / Missing values per column:")
        print(missing_counts[missing_counts > 0])
        
        # 删除气象变量缺失的行
        # Remove rows with missing meteorological variables
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=self.meteorological_vars)
        removed_rows = initial_rows - len(self.df)
        
        if removed_rows > 0:
            print(f"\n删除了 {removed_rows} 行气象数据缺失的记录")
            print(f"Removed {removed_rows} rows with missing meteorological data")
        
        print(f"预处理后数据: {len(self.df)} 行")
        print(f"Data after preprocessing: {len(self.df)} rows")
        
        return self
    
    def normalize_meteorological_data(self):
        """
        对气象数据进行归一化 / Normalize meteorological data
        使用StandardScaler进行标准化 (均值为0，标准差为1)
        Using StandardScaler for standardization (mean=0, std=1)
        
        Returns:
        --------
        pd.DataFrame : 包含归一化后气象数据的DataFrame
        """
        print("\n正在进行气象数据归一化...")
        print("Normalizing meteorological data...")
        
        # 提取气象变量
        # Extract meteorological variables
        meteo_data = self.df[self.meteorological_vars].copy()
        
        # 标准化
        # Standardize
        normalized_data = self.scaler.fit_transform(meteo_data)
        
        # 创建归一化后的DataFrame
        # Create normalized DataFrame
        normalized_columns = [f'{col}_normalized' for col in self.meteorological_vars]
        normalized_df = pd.DataFrame(
            normalized_data,
            columns=normalized_columns,
            index=self.df.index
        )
        
        # 添加到原始数据框
        # Add to original dataframe
        self.df = pd.concat([self.df, normalized_df], axis=1)
        
        print("气象数据归一化完成！")
        print("Meteorological data normalization completed!")
        print(f"\n归一化统计信息 / Normalization statistics:")
        print(f"均值 (Mean): {normalized_df.mean().mean():.6f}")
        print(f"标准差 (Std): {normalized_df.std().mean():.6f}")
        
        return self
    
    def get_normalized_data(self):
        """
        获取归一化后的完整数据 / Get normalized complete data
        
        Returns:
        --------
        pd.DataFrame : 包含原始数据和归一化数据的DataFrame
        """
        return self.df
    
    def save_normalized_data(self, output_file):
        """
        保存归一化后的数据 / Save normalized data
        
        Parameters:
        -----------
        output_file : str
            输出文件路径 / Output file path
        """
        print(f"\n正在保存归一化数据到: {output_file}")
        print(f"Saving normalized data to: {output_file}")
        self.df.to_excel(output_file, index=False)
        print("保存完成！")
        print("Save completed!")
        
    def get_statistics(self):
        """
        获取归一化前后的统计信息 / Get statistics before and after normalization
        
        Returns:
        --------
        dict : 包含统计信息的字典
        """
        stats = {}
        
        # 原始气象数据统计
        # Original meteorological data statistics
        stats['original'] = self.df[self.meteorological_vars].describe()
        
        # 归一化气象数据统计
        # Normalized meteorological data statistics
        normalized_cols = [f'{col}_normalized' for col in self.meteorological_vars]
        if all(col in self.df.columns for col in normalized_cols):
            stats['normalized'] = self.df[normalized_cols].describe()
        
        return stats
    
    def print_statistics(self):
        """打印统计信息 / Print statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("原始气象数据统计 / Original Meteorological Data Statistics")
        print("="*80)
        print(stats['original'])
        
        if 'normalized' in stats:
            print("\n" + "="*80)
            print("归一化后气象数据统计 / Normalized Meteorological Data Statistics")
            print("="*80)
            print(stats['normalized'])


def main():
    """主函数 / Main function"""
    print("="*80)
    print("石家庄市气象数据归一化处理")
    print("Shijiazhuang Meteorological Data Normalization")
    print("="*80)
    
    # 数据文件路径
    # Data file path
    data_file = '石家庄市污染物数据及气象数据.xlsx'
    output_file = '石家庄市污染物数据及气象数据_归一化.xlsx'
    
    # 创建归一化对象并处理
    # Create normalization object and process
    normalizer = MeteorologicalNormalization(data_file)
    
    # 执行归一化流程
    # Execute normalization workflow
    normalizer.load_data()
    normalizer.preprocess_data()
    normalizer.normalize_meteorological_data()
    
    # 打印统计信息
    # Print statistics
    normalizer.print_statistics()
    
    # 保存结果
    # Save results
    normalizer.save_normalized_data(output_file)
    
    print("\n" + "="*80)
    print("处理完成！")
    print("Processing completed!")
    print("="*80)
    print(f"\n归一化后的数据已保存到: {output_file}")
    print(f"Normalized data saved to: {output_file}")
    
    # 显示前几行归一化结果
    # Show first few rows of normalized results
    print("\n前5行归一化数据示例 / First 5 rows of normalized data:")
    normalized_cols = [f'{col}_normalized' for col in normalizer.meteorological_vars]
    print(normalizer.df[['data.1'] + normalized_cols].head())


if __name__ == '__main__':
    main()
