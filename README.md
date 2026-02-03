# SJZ
石家庄市污染物小时级数据及气象小时级数据处理

## 功能 / Features

### 气象归一化处理 / Meteorological Normalization
本项目提供了气象数据归一化处理功能，用于将石家庄市的气象变量标准化处理。

This project provides meteorological data normalization functionality to standardize meteorological variables for Shijiazhuang city.

## 使用方法 / Usage

### 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

### 运行气象归一化 / Run Meteorological Normalization

```bash
python meteorological_normalization.py
```

### 功能说明 / Features Description

气象归一化脚本会：
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
- **输出文件**: `石家庄市污染物数据及气象数据_归一化.xlsx`

归一化方法使用StandardScaler，将数据转换为均值为0、标准差为1的标准正态分布。

The normalization method uses StandardScaler to transform data into a standard normal distribution with mean=0 and standard deviation=1.
