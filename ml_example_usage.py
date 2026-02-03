#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习气象归一化示例
ML-Based Meteorological Normalization Examples

展示如何使用机器学习方法进行气象归一化处理
Demonstrates how to use ML methods for meteorological normalization
"""

from ml_meteorological_normalization import MLMeteorologicalNormalization
import warnings
warnings.filterwarnings('ignore')


def example1_basic_rf():
    """
    示例1: 使用随机森林进行基础气象归一化
    Example 1: Basic meteorological normalization using Random Forest
    """
    print("="*80)
    print("示例 1: 随机森林气象归一化")
    print("Example 1: Random Forest Meteorological Normalization")
    print("="*80)
    
    # 创建归一化对象
    normalizer = MLMeteorologicalNormalization(
        data_file='石家庄市污染物数据及气象数据.xlsx',
        pollutant='pm2.5',
        model_type='rf',
        n_samples=500
    )
    
    # 执行完整流程
    normalizer.load_data()
    normalizer.preprocess_data()
    normalizer.build_model()
    
    # 仅处理1月数据用于演示
    print("\n筛选2025年1月数据用于演示...")
    normalizer.df = normalizer.df[
        (normalizer.df['year'] == 2025) & 
        (normalizer.df['month'] == 1)
    ].reset_index(drop=True)
    
    normalizer.deweather()
    normalizer.plot_comparison(save_path='example1_pm25_rf_comparison.png')
    normalizer.plot_feature_importance(save_path='example1_pm25_rf_importance.png')
    normalizer.save_results('example1_pm25_rf_results.xlsx')
    normalizer.print_summary()
    
    print("\n✓ 示例1完成！")


def example2_xgboost():
    """
    示例2: 使用XGBoost进行气象归一化
    Example 2: Meteorological normalization using XGBoost
    """
    print("\n" + "="*80)
    print("示例 2: XGBoost气象归一化")
    print("Example 2: XGBoost Meteorological Normalization")
    print("="*80)
    
    normalizer = MLMeteorologicalNormalization(
        data_file='石家庄市污染物数据及气象数据.xlsx',
        pollutant='pm2.5',
        model_type='xgb',
        n_samples=500
    )
    
    normalizer.load_data()
    normalizer.preprocess_data()
    normalizer.build_model()
    
    # 筛选1月数据
    normalizer.df = normalizer.df[
        (normalizer.df['year'] == 2025) & 
        (normalizer.df['month'] == 1)
    ].reset_index(drop=True)
    
    normalizer.deweather()
    normalizer.plot_comparison(save_path='example2_pm25_xgb_comparison.png')
    normalizer.print_summary()
    
    print("\n✓ 示例2完成！")


def example3_multiple_pollutants():
    """
    示例3: 对多种污染物进行气象归一化
    Example 3: Meteorological normalization for multiple pollutants
    """
    print("\n" + "="*80)
    print("示例 3: 多污染物气象归一化")
    print("Example 3: Multiple Pollutants Meteorological Normalization")
    print("="*80)
    
    pollutants = ['pm2.5', 'pm10', 'NO2']
    
    for pollutant in pollutants:
        print(f"\n处理污染物: {pollutant.upper()}")
        print(f"Processing pollutant: {pollutant.upper()}")
        
        try:
            normalizer = MLMeteorologicalNormalization(
                data_file='石家庄市污染物数据及气象数据.xlsx',
                pollutant=pollutant,
                model_type='rf',
                n_samples=300
            )
            
            normalizer.load_data()
            normalizer.preprocess_data()
            normalizer.build_model()
            
            # 筛选1月数据
            normalizer.df = normalizer.df[
                (normalizer.df['year'] == 2025) & 
                (normalizer.df['month'] == 1)
            ].reset_index(drop=True)
            
            normalizer.deweather()
            normalizer.plot_comparison(save_path=f'example3_{pollutant}_comparison.png')
            
            # 打印简要统计
            stats = normalizer.get_summary_statistics()
            print(f"原始浓度均值: {stats['original_mean']:.2f}")
            print(f"归一化浓度均值: {stats['normalized_mean']:.2f}")
            print(f"气象贡献均值: {stats['meteo_contribution_mean']:.2f}")
            
        except Exception as e:
            print(f"处理 {pollutant} 时出错: {e}")
    
    print("\n✓ 示例3完成！")


def example4_custom_time_period():
    """
    示例4: 自定义时间段的气象归一化
    Example 4: Custom time period meteorological normalization
    """
    print("\n" + "="*80)
    print("示例 4: 自定义时间段分析")
    print("Example 4: Custom Time Period Analysis")
    print("="*80)
    
    normalizer = MLMeteorologicalNormalization(
        data_file='石家庄市污染物数据及气象数据.xlsx',
        pollutant='pm2.5',
        model_type='rf',
        n_samples=500
    )
    
    normalizer.load_data()
    normalizer.preprocess_data()
    normalizer.build_model()
    
    # 筛选特定日期范围
    normalizer.df = normalizer.df[
        (normalizer.df['datetime'] >= '2025-01-15') & 
        (normalizer.df['datetime'] <= '2025-01-20')
    ].reset_index(drop=True)
    
    print(f"\n分析时间段: 2025-01-15 至 2025-01-20")
    print(f"数据点数: {len(normalizer.df)}")
    
    normalizer.deweather()
    normalizer.plot_comparison(
        start_date='2025-01-15',
        end_date='2025-01-20',
        save_path='example4_custom_period.png'
    )
    normalizer.print_summary()
    
    print("\n✓ 示例4完成！")


def example5_analyze_meteo_contribution():
    """
    示例5: 分析气象贡献的统计特征
    Example 5: Analyze meteorological contribution statistics
    """
    print("\n" + "="*80)
    print("示例 5: 气象贡献深度分析")
    print("Example 5: In-depth Meteorological Contribution Analysis")
    print("="*80)
    
    normalizer = MLMeteorologicalNormalization(
        data_file='石家庄市污染物数据及气象数据.xlsx',
        pollutant='pm2.5',
        model_type='rf',
        n_samples=500
    )
    
    normalizer.load_data()
    normalizer.preprocess_data()
    normalizer.build_model()
    
    # 筛选1月数据
    normalizer.df = normalizer.df[
        (normalizer.df['year'] == 2025) & 
        (normalizer.df['month'] == 1)
    ].reset_index(drop=True)
    
    normalizer.deweather()
    
    # 详细分析气象贡献
    meteo_contrib = normalizer.df[f'{normalizer.pollutant}_meteo_contribution']
    
    print("\n气象贡献详细统计:")
    print(f"最大正贡献（气象促进）: {meteo_contrib.max():.2f}")
    print(f"最大负贡献（气象抑制）: {meteo_contrib.min():.2f}")
    print(f"正贡献天数占比: {(meteo_contrib > 0).sum() / len(meteo_contrib) * 100:.1f}%")
    print(f"负贡献天数占比: {(meteo_contrib < 0).sum() / len(meteo_contrib) * 100:.1f}%")
    
    # 分析气象条件有利和不利时段
    positive_contrib_days = normalizer.df[meteo_contrib > 10]
    negative_contrib_days = normalizer.df[meteo_contrib < -10]
    
    print(f"\n气象条件显著促进污染时段数: {len(positive_contrib_days)}")
    if len(positive_contrib_days) > 0:
        print(f"  平均温度: {positive_contrib_days['temp'].mean():.1f}°C")
        print(f"  平均相对湿度: {positive_contrib_days['rh'].mean():.1f}%")
        print(f"  平均风速: {positive_contrib_days['ws'].mean():.1f}")
    
    print(f"\n气象条件显著抑制污染时段数: {len(negative_contrib_days)}")
    if len(negative_contrib_days) > 0:
        print(f"  平均温度: {negative_contrib_days['temp'].mean():.1f}°C")
        print(f"  平均相对湿度: {negative_contrib_days['rh'].mean():.1f}%")
        print(f"  平均风速: {negative_contrib_days['ws'].mean():.1f}")
    
    print("\n✓ 示例5完成！")


def main():
    """主函数 / Main function"""
    print("机器学习气象归一化示例程序")
    print("ML-Based Meteorological Normalization Examples")
    print()
    
    # 选择要运行的示例
    print("可用示例:")
    print("1. 随机森林气象归一化")
    print("2. XGBoost气象归一化")
    print("3. 多污染物气象归一化")
    print("4. 自定义时间段分析")
    print("5. 气象贡献深度分析")
    print()
    
    # 运行示例1作为演示
    example1_basic_rf()
    
    # 如果需要运行其他示例，取消注释:
    # example2_xgboost()
    # example3_multiple_pollutants()
    # example4_custom_time_period()
    # example5_analyze_meteo_contribution()
    
    print("\n" + "="*80)
    print("所有示例运行完成！")
    print("All examples completed!")
    print("="*80)


if __name__ == '__main__':
    main()
