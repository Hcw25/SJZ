# SJZ
石家庄市污染物小时级数据及气象小时级数据处理

## 功能 / Features

### 1. 机器学习气象归一化处理 (推荐) / ML-Based Meteorological Normalization (Recommended)
**基于随机森林和XGBoost的气象归一化**，用于分离排放强度和气象影响。

ML-based meteorological normalization using Random Forest and XGBoost to separate emission intensity from meteorological effects.

**物理意义 / Physical Meaning:**
- **C_norm (归一化浓度)**: 代表**排放强度**或**污染源贡献** / Represents **emission intensity** or **source contribution**
- **C_observed - C_norm**: 代表**气象因素对浓度波动的贡献绝对值** / Represents **meteorological contribution** to concentration fluctuation

### 2. 标准气象归一化处理 / Standard Meteorological Normalization
基于StandardScaler的简单气象数据标准化处理。

Simple standardization of meteorological variables using StandardScaler.

## 使用方法 / Usage

### 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

### 方法1: 机器学习气象归一化 (推荐) / ML-Based Normalization (Recommended)

#### 快速开始 / Quick Start

```bash
python ml_meteorological_normalization.py
```

#### 程序化使用 / Programmatic Usage

```python
from ml_meteorological_normalization import MLMeteorologicalNormalization

# 创建归一化对象 / Create normalization object
normalizer = MLMeteorologicalNormalization(
    data_file='石家庄市污染物数据及气象数据.xlsx',
    pollutant='pm2.5',      # 可选: pm10, pm2.5, SO2, NO2, CO, O3
    model_type='rf',         # 'rf' (Random Forest) 或 'xgb' (XGBoost)
    n_samples=1000           # 随机抽样次数
)

# 执行完整流程 / Execute full workflow
normalizer.load_data()
normalizer.preprocess_data()
normalizer.build_model()
normalizer.deweather()

# 生成可视化 / Generate visualizations
normalizer.plot_comparison()
normalizer.plot_feature_importance()

# 保存结果 / Save results
normalizer.save_results('output.xlsx')
normalizer.print_summary()
```

#### 查看更多示例 / View More Examples

```bash
python ml_example_usage.py
```

### 方法2: 标准气象归一化 / Standard Normalization

#### 运行标准归一化 / Run Standard Normalization

```bash
python meteorological_normalization.py
```

### 功能说明 / Features Description

#### 机器学习气象归一化 (ML-Based Method)

该方法使用机器学习模型（随机森林或XGBoost）进行真正的气象归一化：

This method uses machine learning models (Random Forest or XGBoost) for true meteorological normalization:

1. **构建预测模型** / Build prediction model:
   - 特征 (Features): 气象变量 + 时间变量
   - 目标 (Target): 污染物观测值
   - 模型评估: R², RMSE, MAE

2. **执行解耦过程** / Perform deweathering:
   - 保持时间特征不变
   - 随机抽样气象数据（默认1000次）
   - 计算预测平均值作为归一化浓度

3. **输出结果** / Output results:
   - **归一化浓度 (C_norm)**: 反映排放强度
   - **气象贡献 (C_obs - C_norm)**: 反映气象影响
   - 对比可视化图表
   - 特征重要性分析

#### 标准气象归一化 (Standard Method)

简单的标准化方法：
1. 加载石家庄市污染物数据及气象数据
2. 对以下气象变量进行标准化归一化处理：
   - sp (气压 / Surface Pressure)
   - temp (温度 / Temperature)
   - wd (风向 / Wind Direction)
   - ws (风速 / Wind Speed)
   - rh (相对湿度 / Relative Humidity)
   - rf (降雨量 / Rainfall)
   - tcc (总云量 / Total Cloud Cover)
3. 输出归一化后的数据到新的Excel文件

The meteorological normalization script will:
1. Load Shijiazhuang pollutant and meteorological data
2. Standardize and normalize the following meteorological variables:
   - sp (Surface Pressure)
   - temp (Temperature)
   - wd (Wind Direction)
   - ws (Wind Speed)
   - rh (Relative Humidity)
   - rf (Rainfall)
   - tcc (Total Cloud Cover)
3. Output normalized data to a new Excel file

### 数据说明 / Data Description

- **输入文件**: `石家庄市污染物数据及气象数据.xlsx`
- **输出文件 (ML方法)**: `{pollutant}_ml_normalized_results.xlsx`
- **输出文件 (标准方法)**: `石家庄市污染物数据及气象数据_归一化.xlsx`
- **可视化图表**: PNG格式对比图和特征重要性图

#### ML方法归一化原理 / ML-Based Normalization Principle

使用机器学习模型学习"气象条件+时间特征"与"污染物浓度"之间的关系。通过随机重采样气象条件并保持时间特征不变，可以消除气象因素的影响，得到的归一化浓度反映真实的排放强度。

Uses machine learning to learn the relationship between "meteorological conditions + temporal features" and "pollutant concentration". By randomly resampling meteorological conditions while keeping temporal features constant, we can eliminate meteorological effects and obtain normalized concentrations that reflect true emission intensity.

**气象贡献的解释 / Interpretation of Meteorological Contribution:**
- **正值 (Positive)**: 气象条件促进污染物累积（不利扩散条件）
- **负值 (Negative)**: 气象条件抑制污染物累积（有利扩散条件）

#### 标准方法说明 / Standard Method Description

标准方法使用StandardScaler，将数据转换为均值为0、标准差为1的标准正态分布。

The standard method uses StandardScaler to transform data into a standard normal distribution with mean=0 and standard deviation=1.

## 模型性能 / Model Performance

以PM2.5为例，使用随机森林模型 / Example with PM2.5 using Random Forest:

- **训练集 R² / Train R²**: 0.974
- **测试集 R² / Test R²**: 0.904
- **测试集 RMSE / Test RMSE**: 11.8 μg/m³

**特征重要性 Top 5 / Feature Importance Top 5:**
1. day_julian (日序数): 28.6%
2. rh (相对湿度): 24.2%
3. sp (气压): 10.5%
4. temp (温度): 9.1%
5. day (日): 6.8%

## 示例结果 / Example Results

运行ML气象归一化后，将生成以下图表：
After running ML meteorological normalization, the following plots will be generated:

1. **对比图 (Comparison Plot)**: 显示原始观测值、归一化浓度、气象贡献的时间序列
2. **特征重要性图 (Feature Importance)**: 显示各特征对预测的贡献度

参见生成的PNG文件 / See generated PNG files for visual results.
