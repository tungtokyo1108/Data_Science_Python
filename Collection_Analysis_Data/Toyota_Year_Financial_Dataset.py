# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:53:18 2025

@author: TungDang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Enhanced styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


years = list(range(2015, 2026))
        
# Core financial dataset
raw_data = {
    'Year': years,
    'Net_Revenue_T': [27.23, 28.40, 29.38, 29.88, 30.23, 27.21, 31.38, 37.15, 37.15, 45.10, 48.04],
    'Operating_Income_T': [2.75, 2.85, 1.99, 2.40, 2.46, 2.44, 2.99, 2.72, 2.72, 5.35, 4.80],
    'Net_Income_T': [2.17, 2.31, 1.83, 2.49, 2.07, 2.04, 2.25, 2.85, 2.45, 4.95, 4.77],
    'Operating_Margin': [10.1, 10.0, 6.8, 8.0, 8.1, 9.0, 9.5, 7.3, 7.3, 11.9, 10.0],
    'Net_Margin': [8.0, 8.1, 6.2, 8.3, 6.8, 7.5, 7.2, 7.7, 6.6, 11.0, 9.9],
    'Revenue_Growth': [-8.9, 4.3, 3.4, 1.7, 1.2, -10.0, 15.3, 18.4, 0.0, 21.4, 6.5],
    'Total_Assets_T': [47.4, 49.8, 62.3, 67.7, 74.3, 74.3, 90.1, 90.1, 90.1, 93.6, 96.8],
    'Total_Equity_T': [18.7, 19.5, 24.3, 27.2, 29.3, 28.3, 35.2, 35.9, 34.2, 35.9, 38.8],
    'Cash_Equivalents_T': [3.8, 4.2, 6.1, 6.8, 7.2, 8.3, 11.6, 12.5, 11.9, 15.9, 16.2],
    'ROE': [11.5, 10.2, 11.5, 11.5, 10.2, 9.0, 10.2, 11.5, 9.0, 15.8, 13.6],
    'ROA': [4.2, 3.9, 4.4, 4.4, 3.9, 3.5, 3.9, 4.4, 3.5, 6.0, 5.2],
    'Debt_to_Equity': [0.95, 1.06, 0.98, 1.02, 1.19, 1.26, 1.05, 1.02, 1.08, 0.98, 1.05],
    'Current_Ratio': [1.15, 1.06, 1.09, 1.10, 1.19, 1.26, 1.07, 1.15, 1.12, 1.18, 1.22],
    'Vehicle_Sales_M': [10.15, 10.23, 10.35, 10.59, 10.74, 9.53, 10.50, 10.38, 11.23, 11.13, 9.50],
    'Market_Share': [10.3, 10.4, 10.6, 10.8, 10.9, 10.5, 10.8, 10.5, 10.7, 11.1, 10.8],
    'Revenue_per_Vehicle': [2.68, 2.78, 2.84, 2.82, 2.81, 2.86, 2.99, 3.58, 3.31, 4.05, 5.06],
    'Operating_Income_per_Vehicle': [271, 279, 192, 227, 229, 256, 285, 262, 242, 481, 505],
    'RD_Spending_B': [1055, 1085, 1124, 1406, 1363, 1326, 1158, 1245, 1198, 1200, 1326],
    'RD_Revenue_Ratio': [3.9, 3.8, 3.8, 4.7, 4.5, 4.9, 3.7, 3.3, 3.2, 2.7, 2.8],
    'Patents_Filed': [2341, 2456, 2598, 2721, 2834, 2753, 2891, 3045, 3187, 3298, 3421],
    'Patents_per_RD_B': [2.22, 2.26, 2.31, 1.94, 2.08, 2.08, 2.50, 2.45, 2.66, 2.75, 2.58],
    'Electrified_Percentage': [15.6, 16.8, 18.3, 20.0, 23.2, 28.5, 30.2, 34.1, 38.1, 42.9, 54.8],
    'BEV_Sales_M': [0.00, 0.01, 0.02, 0.04, 0.07, 0.12, 0.18, 0.24, 0.36, 0.48, 0.63],
    'CO2_Emissions_M': [30.2, 29.8, 29.1, 28.7, 28.5, 28.4, 27.2, 25.8, 24.1, 22.6, 20.8],
    'Renewable_Energy': [18, 19, 21, 22, 23, 23, 26, 31, 37, 42, 48],
    'FS_Revenue_B': [2240, 2180, 2350, 2420, 2485, 2380, 2520, 2665, 2744, 2890, 3120],
    'FS_Operating_Income_B': [287, 264, 298, 315, 321, 285, 336, 358, 329, 147, 160],
    'Employee_Count_K': [348, 352, 369, 370, 372, 366, 372, 375, 381, 384, 388],
    'Revenue_per_Employee': [78.3, 80.7, 79.6, 80.8, 81.3, 74.3, 84.3, 99.1, 97.5, 117.4, 123.8],
    'EV_Investment_B': [180, 210, 250, 290, 330, 350, 380, 410, 440, 450, 520],
    'Digital_Revenue_B': [45, 62, 85, 125, 180, 220, 285, 380, 495, 680, 920]
}

data = pd.DataFrame(raw_data)
data.set_index('Year', inplace=True)


"""
Calculate advanced financial metrics for deeper PCA insights
"""
# Financial efficiency metrics
data['Asset_Turnover'] = data['Net_Revenue_T'] / data['Total_Assets_T']
data['Equity_Multiplier'] = data['Total_Assets_T'] / data['Total_Equity_T']
data['Financial_Leverage'] = data['Total_Assets_T'] / data['Total_Equity_T']

# Profitability evolution metrics
data['EBIT_Margin'] = data['Operating_Margin']  # Proxy
data['Net_Profit_Margin'] = data['Net_Margin']
data['Return_on_Capital'] = data['Operating_Income_T'] / data['Total_Assets_T'] * 100

# Growth and transformation metrics
data['Innovation_Intensity'] = data['RD_Spending_B'] / data['Employee_Count_K']
data['Digital_Transformation_Rate'] = data['Digital_Revenue_B'] / data['Net_Revenue_T'] * 1000
data['Sustainability_Index'] = (data['Electrified_Percentage'] + data['Renewable_Energy']) / 2
data['Green_Investment_Rate'] = data['EV_Investment_B'] / data['Net_Revenue_T'] * 1000

# Operational efficiency metrics
data['Capital_Efficiency'] = data['Operating_Income_T'] / data['Total_Assets_T'] * 100
data['Cash_Conversion_Efficiency'] = data['Cash_Equivalents_T'] / data['Net_Revenue_T'] * 100
data['Employee_Productivity'] = data['Operating_Income_T'] / data['Employee_Count_K'] * 1000

# Market positioning metrics
data['Premium_Positioning'] = data['Revenue_per_Vehicle'] / data['Market_Share']
data['Innovation_ROI'] = data['Patents_Filed'] / data['RD_Spending_B'] * 1000
data['Market_Value_Creation'] = data['Market_Share'] * data['Operating_Margin']

# Risk and stability metrics
data['Financial_Stability'] = 1 / (1 + data['Debt_to_Equity'])
data['Liquidity_Strength'] = data['Current_Ratio'] * data['Cash_Equivalents_T']
data['Business_Resilience'] = data['Operating_Margin'] * data['Current_Ratio']


"""
Perform comprehensive PCA analysis with multiple scaling approaches
"""

# Prepare data
feature_data = data.select_dtypes(include=[np.number])
feature_names = feature_data.columns.tolist()

# Try different scaling approaches for robustness
scalers = {
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler()
}

pca_results = {}

for scaler_name, scaler in scalers.items():
    print(f"\n📊 Analyzing with {scaler_name}...")
    
    # Scale data
    scaled_data = scaler.fit_transform(feature_data)
    
    # Perform PCA
    pca = PCA()
    pca_components = pca.fit_transform(scaled_data)
    
    # Store results
    pca_results[scaler_name] = {
        'model': pca,
        'components': pca_components,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'loadings': pca.components_,
        'scaled_data': scaled_data
    }
    
    # Key insights
    n_80 = np.argmax(pca.explained_variance_ratio_.cumsum() >= 0.8) + 1
    n_95 = np.argmax(pca.explained_variance_ratio_.cumsum() >= 0.95) + 1
    
    print(f"   📈 Components for 80% variance: {n_80}")
    print(f"   📈 Components for 95% variance: {n_95}")
    print(f"   📊 PC1 variance explained: {pca.explained_variance_ratio_[0]*100:.1f}%")

# Use StandardScaler results as primary
pca_model = pca_results['StandardScaler']['model']
pca_results = pca_results['StandardScaler']
scaled_data = pca_results['StandardScaler']['scaled_data']



"""
Deep analysis of principal components from financial perspective
"""

def interpret_component_business_meaning(component_idx, loadings):
    """
    Interpret the business meaning of each principal component
    """
    # Get top contributors
    top_loadings = loadings.abs().nlargest(10)
    
    # Define business themes based on metric categories
    financial_performance = ['Net_Revenue_T', 'Operating_Income_T', 'Net_Income_T', 'ROE', 'ROA', 'Operating_Margin']
    business_scale = ['Total_Assets_T', 'Vehicle_Sales_M', 'Employee_Count_K', 'Market_Share']
    innovation = ['RD_Spending_B', 'Patents_Filed', 'Digital_Revenue_B', 'Innovation_ROI']
    sustainability = ['Electrified_Percentage', 'BEV_Sales_M', 'Renewable_Energy', 'CO2_Emissions_M']
    efficiency = ['Revenue_per_Vehicle', 'Revenue_per_Employee', 'Asset_Turnover', 'Capital_Efficiency']
    financial_strength = ['Cash_Equivalents_T', 'Current_Ratio', 'Financial_Stability', 'Liquidity_Strength']
    
    # Analyze which themes dominate this component
    theme_scores = {
        'Financial Performance': sum(abs(loadings[m]) for m in financial_performance if m in loadings.index),
        'Business Scale': sum(abs(loadings[m]) for m in business_scale if m in loadings.index),
        'Innovation & Technology': sum(abs(loadings[m]) for m in innovation if m in loadings.index),
        'Sustainability Transformation': sum(abs(loadings[m]) for m in sustainability if m in loadings.index),
        'Operational Efficiency': sum(abs(loadings[m]) for m in efficiency if m in loadings.index),
        'Financial Strength': sum(abs(loadings[m]) for m in financial_strength if m in loadings.index)
    }
    
    # Find dominant theme
    dominant_theme = max(theme_scores, key=theme_scores.get)
    
    # Component-specific interpretations
    interpretations = {
        0: "Business Scale & Financial Size - Overall corporate magnitude and market presence",
        1: "Profitability & Efficiency - Operating performance and margin management",
        2: "Innovation & Transformation - R&D investment and technological advancement",
        3: "Sustainability & Future Strategy - Environmental transformation and electrification",
        4: "Financial Resilience - Liquidity, stability, and risk management"
    }
    
    base_interpretation = interpretations.get(component_idx, f"Business Factor {component_idx+1}")
    return f"{base_interpretation} (Dominated by: {dominant_theme})"


loadings = pca_results['loadings']
variance_explained = pca_results['explained_variance_ratio']

# Analyze first 5 components (typically capture 90%+ variance)
component_analysis = {}

for i in range(min(5, len(variance_explained))):
    print(f"\n🔍 PRINCIPAL COMPONENT {i+1} ANALYSIS")
    print(f"   Variance Explained: {variance_explained[i]*100:.1f}%")
    print(f"   Cumulative Variance: {np.sum(variance_explained[:i+1])*100:.1f}%")
    
    # Get loadings for this component
    component_loadings = pd.Series(loadings[i], index=feature_names)
    
    # Identify highest positive and negative loadings
    top_positive = component_loadings.nlargest(8)
    top_negative = component_loadings.nsmallest(8)
    
    print(f"\n   💡 TOP POSITIVE CONTRIBUTORS:")
    for metric, loading in top_positive.items():
        if abs(loading) > 0.15:  # Significance threshold
            print(f"      • {metric}: {loading:.3f}")
    
    print(f"\n   ⚠️  TOP NEGATIVE CONTRIBUTORS:")
    for metric, loading in top_negative.items():
        if abs(loading) < -0.15:  # Significance threshold
            print(f"      • {metric}: {loading:.3f}")
    
    # Business interpretation
    component_interpretation = interpret_component_business_meaning(i, component_loadings)
    
    component_analysis[f'PC{i+1}'] = {
        'variance_explained': variance_explained[i],
        'loadings': component_loadings,
        'top_positive': top_positive,
        'top_negative': top_negative,
        'business_meaning': component_interpretation
    }
    
    print(f"\n   🎯 BUSINESS INTERPRETATION: {component_interpretation}")

business_factors = component_analysis



"""
Analyze how Toyota's financial position evolves in PCA space over time
"""

def identify_business_phases(temporal_analysis):
    """
    Identify distinct business transformation phases based on PCA evolution
    """
    print(f"\n🔄 BUSINESS TRANSFORMATION PHASES:")
    
    # Use PC1 and PC2 for phase identification
    pc1_scores = temporal_analysis['PC1']['scores']
    pc2_scores = temporal_analysis['PC2']['scores']
    
    # Define phases based on major component shifts
    #years = self.data.index
    
    # Simple phase identification based on PC1 trend changes
    pc1_changes = np.diff(pc1_scores)
    phase_boundaries = []
    
    for i, change in enumerate(pc1_changes):
        if abs(change) > np.std(pc1_changes) * 1.5:
            phase_boundaries.append(i + 1)
    
    # Define business phases
    phases = {
        '2015-2017': 'Consolidation Phase - Stability and incremental growth',
        '2018-2020': 'Disruption & Adaptation - Market challenges and strategic pivots',
        '2021-2023': 'Transformation Phase - Major strategic shifts and investments',
        '2024-2025': 'Premium Positioning - Value-driven growth and market leadership'
    }
    
    for period, description in phases.items():
        print(f"   📅 {period}: {description}")


components = pca_results['components']
years = data.index

# Analyze trends in each principal component
temporal_analysis = {}

for i in range(min(5, components.shape[1])):
    pc_scores = components[:, i]
    
    # Calculate trend
    trend_slope, trend_intercept, r_value, p_value, std_err = stats.linregress(range(len(pc_scores)), pc_scores)
    
    # Identify periods of significant change
    changes = np.diff(pc_scores)
    significant_changes = np.where(np.abs(changes) > np.std(changes) * 1.5)[0]
    
    # Business cycle analysis
    pc_volatility = np.std(pc_scores)
    pc_range = np.max(pc_scores) - np.min(pc_scores)
    
    temporal_analysis[f'PC{i+1}'] = {
        'trend_slope': trend_slope,
        'trend_significance': p_value,
        'r_squared': r_value**2,
        'volatility': pc_volatility,
        'range': pc_range,
        'significant_changes': significant_changes,
        'scores': pc_scores
    }
    
    print(f"\n🔍 PC{i+1} TEMPORAL ANALYSIS:")
    print(f"   📈 Trend: {'Positive' if trend_slope > 0 else 'Negative'} ({trend_slope:.3f}/year)")
    print(f"   📊 Trend Significance: {p_value:.3f} (R² = {r_value**2:.3f})")
    print(f"   📉 Volatility: {pc_volatility:.3f}")
    
    if len(significant_changes) > 0:
        change_years = [years[i+1] for i in significant_changes]
        print(f"   ⚡ Significant Changes: {change_years}")

# Identify business transformation phases
identify_business_phases(temporal_analysis)


"""
Analyze financial risks and opportunities through PCA lens
"""

components = pca_results['components']
loadings = pca_results['loadings']

# Risk analysis based on component volatility and loadings
risk_factors = {}
opportunity_factors = {}

for i in range(min(3, components.shape[1])):
    pc_volatility = np.std(components[:, i])
    pc_loadings = pd.Series(loadings[i], index=feature_names)
    
    # Identify risk indicators (high volatility with negative financial impact)
    risk_metrics = pc_loadings[pc_loadings < -0.2].abs().sort_values(ascending=False)
    opportunity_metrics = pc_loadings[pc_loadings > 0.2].sort_values(ascending=False)
    
    risk_factors[f'PC{i+1}'] = {
        'volatility': pc_volatility,
        'risk_metrics': risk_metrics.head(5),
        'current_position': components[-1, i]  # 2025 position
    }
    
    opportunity_factors[f'PC{i+1}'] = {
        'opportunity_metrics': opportunity_metrics.head(5),
        'growth_trend': np.polyfit(range(len(components[:, i])), components[:, i], 1)[0]
    }


"""
Create comprehensive PCA visualization dashboard
"""

# Create main figure with multiple subplots
fig = plt.figure(figsize=(24, 20))

# 1. Variance Explained Analysis
ax1 = plt.subplot(4, 4, 1)
variance_ratios = pca_results['explained_variance_ratio'][:10]
plt.bar(range(1, len(variance_ratios) + 1), variance_ratios, 
        color='steelblue', alpha=0.8, edgecolor='navy')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained Ratio')
plt.title('Individual Component Variance', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3)

# 2. Cumulative Variance
ax2 = plt.subplot(4, 4, 2)
cumvar = pca_results['cumulative_variance'][:10]
plt.plot(range(1, len(cumvar) + 1), cumvar, 'ro-', linewidth=3, markersize=8)
plt.axhline(y=0.8, color='green', linestyle='--', linewidth=2, label='80% Threshold')
plt.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95% Threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 3. PC1 vs PC2 Financial Evolution
ax3 = plt.subplot(4, 4, 3)
components = pca_results['components']
years = data.index

# Create trajectory plot
plt.plot(components[:, 0], components[:, 1], 'b-', linewidth=2, alpha=0.6, label='Trajectory')
scatter = plt.scatter(components[:, 0], components[:, 1], 
                    c=years, cmap='viridis', s=120, alpha=0.8, edgecolors='black')

# Add year labels
for i, year in enumerate(years):
    plt.annotate(str(year), (components[i, 0], components[i, 1]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

plt.xlabel(f'PC1 - Business Scale ({variance_ratios[0]*100:.1f}%)')
plt.ylabel(f'PC2 - Operational Performance ({variance_ratios[1]*100:.1f}%)')
plt.title('Financial Evolution Trajectory', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3)
plt.colorbar(scatter, label='Year')

# 4. PC1 Loadings Analysis
ax4 = plt.subplot(4, 4, 4)
pc1_loadings = pd.Series(pca_results['loadings'][0], index=feature_names)
top_pc1 = pc1_loadings.abs().nlargest(12)
colors = ['red' if pc1_loadings[idx] < 0 else 'blue' for idx in top_pc1.index]

plt.barh(range(len(top_pc1)), [pc1_loadings[idx] for idx in top_pc1.index], 
        color=colors, alpha=0.7)
plt.yticks(range(len(top_pc1)), [idx.replace('_', ' ') for idx in top_pc1.index])
plt.xlabel('Loading Value')
plt.title('PC1 - Top Contributing Factors', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3)

# 5. PC2 Loadings Analysis
ax5 = plt.subplot(4, 4, 5)
pc2_loadings = pd.Series(pca_results['loadings'][1], index=feature_names)
top_pc2 = pc2_loadings.abs().nlargest(12)
colors = ['red' if pc2_loadings[idx] < 0 else 'green' for idx in top_pc2.index]

plt.barh(range(len(top_pc2)), [pc2_loadings[idx] for idx in top_pc2.index], 
        color=colors, alpha=0.7)
plt.yticks(range(len(top_pc2)), [idx.replace('_', ' ') for idx in top_pc2.index])
plt.xlabel('Loading Value')
plt.title('PC2 - Top Contributing Factors', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3)

# 6. Component Evolution Over Time
ax6 = plt.subplot(4, 4, 6)
plt.plot(years, components[:, 0], 'o-', linewidth=3, markersize=8, label='PC1 (Scale)', color='blue')
plt.plot(years, components[:, 1], 's-', linewidth=3, markersize=8, label='PC2 (Performance)', color='green')
plt.plot(years, components[:, 2], '^-', linewidth=3, markersize=8, label='PC3 (Innovation)', color='red')
plt.xlabel('Year')
plt.ylabel('Component Score')
plt.title('Principal Component Evolution', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 7. 3D Component Space
ax7 = plt.subplot(4, 4, 7, projection='3d')
scatter3d = ax7.scatter(components[:, 0], components[:, 1], components[:, 2],
                      c=years, cmap='plasma', s=100, alpha=0.8)
ax7.set_xlabel('PC1 (Scale)')
ax7.set_ylabel('PC2 (Performance)')
ax7.set_zlabel('PC3 (Innovation)')
ax7.set_title('3D Financial Space', fontweight='bold', fontsize=12)

# 8. Financial Health Radar Chart
ax8 = plt.subplot(4, 4, 8, projection='polar')

# Select key financial metrics for radar
health_metrics = ['ROE', 'ROA', 'Operating_Margin', 'Current_Ratio', 
                 'Revenue_per_Vehicle', 'Market_Share']
latest_values = []

for metric in health_metrics:
    if metric in data.columns:
        # Normalize to 0-1 scale for radar chart
        max_val = data[metric].max()
        min_val = data[metric].min()
        normalized = (data[metric].iloc[-1] - min_val) / (max_val - min_val)
        latest_values.append(normalized)

# Create radar chart
angles = np.linspace(0, 2*np.pi, len(health_metrics), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
latest_values = latest_values + [latest_values[0]]  # Complete the circle

ax8.plot(angles, latest_values, 'o-', linewidth=2, color='blue', alpha=0.7)
ax8.fill(angles, latest_values, alpha=0.3, color='blue')
ax8.set_xticks(angles[:-1])
ax8.set_xticklabels([m.replace('_', ' ') for m in health_metrics])
ax8.set_title('Current Financial Health Profile', fontweight='bold', fontsize=12)

# 9. Loadings Correlation Heatmap
ax9 = plt.subplot(4, 4, 9)
loadings_df = pd.DataFrame(pca_results['loadings'][:4].T, 
                         columns=['PC1', 'PC2', 'PC3', 'PC4'],
                         index=feature_names)

# Select top contributing features
top_features = []
for i in range(4):
    pc_loadings = loadings_df[f'PC{i+1}'].abs()
    top_features.extend(pc_loadings.nlargest(5).index.tolist())
top_features = list(set(top_features))[:20]  # Limit for readability

sns.heatmap(loadings_df.loc[top_features], annot=True, cmap='RdBu_r', center=0, 
           fmt='.2f', ax=ax9, cbar=True)
ax9.set_title('Feature-Component Loadings', fontweight='bold', fontsize=12)

# 10. Financial Transformation Timeline
ax10 = plt.subplot(4, 4, 10)

# Key transformation metrics
transformation_metrics = ['Electrified_Percentage', 'Digital_Revenue_B', 'Revenue_per_Vehicle']

for metric in transformation_metrics:
    if metric in data.columns:
        normalized_values = (data[metric] - data[metric].min()) / (data[metric].max() - data[metric].min())
        plt.plot(years, normalized_values, 'o-', linewidth=2, markersize=6, 
                label=metric.replace('_', ' ').replace('B', '(B)').replace('M', '(M)'))

plt.xlabel('Year')
plt.ylabel('Normalized Value (0-1)')
plt.title('Business Transformation Indicators', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 11. Component Contribution by Business Area
ax11 = plt.subplot(4, 4, 11)

# Define business areas
business_areas = {
    'Financial Performance': ['Net_Revenue_T', 'Operating_Income_T', 'ROE', 'ROA'],
    'Innovation': ['RD_Spending_B', 'Patents_Filed', 'Digital_Revenue_B'],
    'Sustainability': ['Electrified_Percentage', 'Renewable_Energy', 'CO2_Emissions_M'],
    'Efficiency': ['Revenue_per_Vehicle', 'Revenue_per_Employee', 'Asset_Turnover']
}

# Calculate average loading magnitude by business area for PC1
pc1_loadings = pd.Series(pca_results['loadings'][0], index=feature_names)
area_contributions = {}

for area, metrics in business_areas.items():
    area_loading = np.mean([abs(pc1_loadings[m]) for m in metrics if m in pc1_loadings.index])
    area_contributions[area] = area_loading

plt.bar(area_contributions.keys(), area_contributions.values(), 
        color=['steelblue', 'orange', 'green', 'red'], alpha=0.7)
plt.ylabel('Average Loading Magnitude')
plt.title('PC1 Contribution by Business Area', fontweight='bold', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 12. Financial Risk-Return Analysis
ax12 = plt.subplot(4, 4, 12)

# Use PC1 (scale) vs PC2 (performance) for risk-return analysis
pc1_scores = components[:, 0]
pc2_scores = components[:, 1]

# Calculate period-over-period changes as "risk"
pc1_volatility = np.std(np.diff(pc1_scores))
pc2_volatility = np.std(np.diff(pc2_scores))

# Plot risk-return by year
for i, year in enumerate(years):
    if i > 0:
        pc1_return = pc1_scores[i] - pc1_scores[i-1]
        pc2_return = pc2_scores[i] - pc2_scores[i-1]
        plt.scatter(abs(pc1_return), pc2_return, s=100, alpha=0.7, 
                  label=str(year) if i % 2 == 0 else "")
        plt.annotate(str(year), (abs(pc1_return), pc2_return), 
                   xytext=(3, 3), textcoords='offset points', fontsize=9)

plt.xlabel('PC1 Change Magnitude (Risk)')
plt.ylabel('PC2 Change (Performance)')
plt.title('Financial Risk-Return Analysis', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3)

# 13-16: Additional detailed analysis plots
# Component stability analysis
ax13 = plt.subplot(4, 4, 13)
component_stabilities = []
for i in range(min(5, components.shape[1])):
    stability = 1 / (1 + np.std(np.diff(components[:, i])))
    component_stabilities.append(stability)

plt.bar(range(1, len(component_stabilities) + 1), component_stabilities, 
        color='purple', alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Stability Score')
plt.title('Component Stability Analysis', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3)

# Business cycle analysis
ax14 = plt.subplot(4, 4, 14)

# Define business cycle phases based on key metrics
cycle_indicator = (data['Revenue_Growth'] + data['Operating_Margin']) / 2
plt.plot(years, cycle_indicator, 'o-', linewidth=3, markersize=8, color='purple')
plt.axhline(y=cycle_indicator.mean(), color='red', linestyle='--', 
           label=f'Average ({cycle_indicator.mean():.1f})')
plt.xlabel('Year')
plt.ylabel('Business Cycle Indicator')
plt.title('Business Cycle Analysis', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Future projection based on PCA trends
ax15 = plt.subplot(4, 4, 15)

# Project PC1 and PC2 trends
for i in range(2):
    pc_scores = components[:, i]
    # Fit linear trend
    trend = np.polyfit(range(len(pc_scores)), pc_scores, 1)
    future_years = list(range(2026, 2030))
    future_values = [trend[0] * (len(pc_scores) + j) + trend[1] for j in range(4)]
    
    # Plot historical and projected
    plt.plot(years, pc_scores, 'o-', linewidth=2, markersize=6, 
            label=f'PC{i+1} Historical')
    plt.plot(future_years, future_values, '--', linewidth=2, alpha=0.7,
            label=f'PC{i+1} Projected')

plt.xlabel('Year')
plt.ylabel('Component Score')
plt.title('PCA-Based Financial Projection', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Strategic positioning map
ax16 = plt.subplot(4, 4, 16)

# Create strategic quadrants based on PC1 and PC2
pc1_mean = np.mean(components[:, 0])
pc2_mean = np.mean(components[:, 1])

# Plot trajectory with quadrant analysis
plt.axhline(y=pc2_mean, color='gray', linestyle='-', alpha=0.5)
plt.axvline(x=pc1_mean, color='gray', linestyle='-', alpha=0.5)

# Color code by time period
colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
for i, (year, color) in enumerate(zip(years, colors)):
    plt.scatter(components[i, 0], components[i, 1], 
               c=[color], s=120, alpha=0.8, edgecolors='black')
    if i % 2 == 0:  # Label every other year for clarity
        plt.annotate(str(year), (components[i, 0], components[i, 1]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

# Add quadrant labels
plt.text(pc1_mean + 0.5, pc2_mean + 0.5, 'High Scale\nHigh Performance', 
        ha='center', va='center', fontsize=10, fontweight='bold', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
plt.text(pc1_mean - 0.5, pc2_mean + 0.5, 'Low Scale\nHigh Performance', 
        ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
plt.text(pc1_mean + 0.5, pc2_mean - 0.5, 'High Scale\nLow Performance', 
        ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
plt.text(pc1_mean - 0.5, pc2_mean - 0.5, 'Low Scale\nLow Performance', 
        ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

plt.xlabel('PC1 - Business Scale')
plt.ylabel('PC2 - Operational Performance')
plt.title('Strategic Positioning Map', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()



























