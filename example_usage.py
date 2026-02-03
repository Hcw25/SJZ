#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气象归一化处理示例 / Meteorological Normalization Example

这个脚本展示了如何使用MeteorologicalNormalization类进行自定义处理
This script demonstrates how to use the MeteorologicalNormalization class for custom processing
"""

from meteorological_normalization import MeteorologicalNormalization
import pandas as pd


def example_basic_usage():
    """基础使用示例 / Basic usage example"""
    print("=" * 80)
    print("示例 1: 基础使用 / Example 1: Basic Usage")
    print("=" * 80)
    
    # 创建归一化对象
    # Create normalization object
    normalizer = MeteorologicalNormalization('石家庄市污染物数据及气象数据.xlsx')
    
    # 执行归一化
    # Execute normalization
    normalizer.load_data().preprocess_data().normalize_meteorological_data()
    
    # 获取归一化后的数据
    # Get normalized data
    normalized_df = normalizer.get_normalized_data()
    
    print(f"\n归一化后的数据形状: {normalized_df.shape}")
    print(f"Normalized data shape: {normalized_df.shape}")


def example_get_statistics():
    """获取统计信息示例 / Get statistics example"""
    print("\n" + "=" * 80)
    print("示例 2: 获取统计信息 / Example 2: Get Statistics")
    print("=" * 80)
    
    normalizer = MeteorologicalNormalization('石家庄市污染物数据及气象数据.xlsx')
    normalizer.load_data().preprocess_data().normalize_meteorological_data()
    
    # 打印统计信息
    # Print statistics
    normalizer.print_statistics()


def example_analyze_specific_period():
    """分析特定时期数据示例 / Analyze specific period example"""
    print("\n" + "=" * 80)
    print("示例 3: 分析特定时期数据 / Example 3: Analyze Specific Period")
    print("=" * 80)
    
    normalizer = MeteorologicalNormalization('石家庄市污染物数据及气象数据.xlsx')
    normalizer.load_data().preprocess_data().normalize_meteorological_data()
    
    df = normalizer.get_normalized_data()
    
    # 分析2025年1月的数据
    # Analyze January 2025 data
    january_data = df[(df['year'] == 2025) & (df['month'] == 1)]
    
    print(f"\n2025年1月数据量: {len(january_data)} 条记录")
    print(f"January 2025 data: {len(january_data)} records")
    
    # 显示归一化后的温度统计
    # Show normalized temperature statistics
    print("\n归一化温度统计 / Normalized Temperature Statistics:")
    print(january_data['temp_normalized'].describe())


def example_compare_normalization():
    """比较归一化前后数据示例 / Compare before and after normalization example"""
    print("\n" + "=" * 80)
    print("示例 4: 比较归一化前后 / Example 4: Compare Before and After")
    print("=" * 80)
    
    normalizer = MeteorologicalNormalization('石家庄市污染物数据及气象数据.xlsx')
    normalizer.load_data().preprocess_data().normalize_meteorological_data()
    
    df = normalizer.get_normalized_data()
    
    # 选择前10条记录进行比较
    # Select first 10 records for comparison
    sample = df.head(10)
    
    print("\n原始温度 vs 归一化温度 / Original Temperature vs Normalized Temperature:")
    comparison = sample[['data.1', 'temp', 'temp_normalized']]
    print(comparison.to_string(index=False))
    
    print("\n原始气压 vs 归一化气压 / Original Pressure vs Normalized Pressure:")
    comparison = sample[['data.1', 'sp', 'sp_normalized']]
    print(comparison.to_string(index=False))


def main():
    """主函数 / Main function"""
    print("气象归一化处理示例程序")
    print("Meteorological Normalization Example Program")
    print()
    
    # 运行所有示例
    # Run all examples
    example_basic_usage()
    example_get_statistics()
    example_analyze_specific_period()
    example_compare_normalization()
    
    print("\n" + "=" * 80)
    print("所有示例运行完成！")
    print("All examples completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
