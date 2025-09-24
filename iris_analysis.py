import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.df = None
        
    def load_iris_dataset(self):
        """Load and prepare the Iris dataset"""
        try:
            iris = load_iris()
            self.data = iris
            self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
            self.df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
            print("✅ Iris dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def basic_analysis(self):
        """Perform basic data analysis"""
        if self.df is None:
            print("❌ No data loaded!")
            return
            
        print("\n" + "="*50)
        print("BASIC DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print("\n📊 Dataset Info:")
        print(self.df.info())
        
        # Descriptive statistics
        print("\n📈 Descriptive Statistics:")
        print(self.df.describe())
        
        # Check for missing values
        print("\n🔍 Missing Values:")
        missing_data = self.df.isnull().sum()
        print(missing_data[missing_data > 0] if missing_data.sum() > 0 else "No missing values found!")
        
        # Data types
        print("\n📋 Data Types:")
        print(self.df.dtypes)
    
    def group_analysis(self):
        """Analyze data by species groups"""
        if self.df is None:
            print("❌ No data loaded!")
            return
            
        print("\n" + "="*50)
        print("GROUP ANALYSIS BY SPECIES")
        print("="*50)
        
        # Group by species and calculate statistics
        grouped = self.df.groupby('species')
        
        print("\n📊 Summary statistics by species:")
        for column in self.df.columns[:-1]:  # Exclude species column
            print(f"\n🌸 {column.upper()} by Species:")
            species_stats = grouped[column].agg(['mean', 'median', 'std', 'min', 'max'])
            print(species_stats)
            
            # Identify patterns
            max_species = species_stats['mean'].idxmax()
            min_species = species_stats['mean'].idxmin()
            print(f"   Highest average: {max_species} ({species_stats.loc[max_species, 'mean']:.2f})")
            print(f"   Lowest average: {min_species} ({species_stats.loc[min_species, 'mean']:.2f})")
    
    def create_visualizations(self):
        """Create the required visualizations"""
        if self.df is None:
            print("❌ No data loaded!")
            return
            
        print("\n" + "="*50)
        print("DATA VISUALIZATION")
        print("="*50)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Iris Dataset Analysis - Comprehensive Visualizations', fontsize=16, fontweight='bold')
        
        # 1. Line Chart - Trends over time (simulated using index as time)
        self._create_line_chart(axes[0, 0])
        
        # 2. Bar Chart - Comparison across categories
        self._create_bar_chart(axes[0, 1])
        
        # 3. Histogram - Distribution of numerical column
        self._create_histogram(axes[1, 0])
        
        # 4. Scatter Plot - Relationship between two numerical columns
        self._create_scatter_plot(axes[1, 1])
        
        plt.tight_layout()
        plt.show()
        
        # Additional advanced visualizations
        self._create_advanced_visualizations()
    
    def _create_line_chart(self, ax):
        """Line chart showing trends (using index as pseudo-time)"""
        # Since Iris dataset doesn't have time series, we'll use index as pseudo-time
        # and show how measurements vary across observations
        features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        
        # Create a sample trend by sorting one feature to simulate time series
        temp_df = self.df.sort_values('sepal length (cm)').reset_index(drop=True)
        
        for feature in features:
            ax.plot(temp_df.index, temp_df[feature], label=feature, linewidth=2, alpha=0.7)
        
        ax.set_title('📈 Feature Trends Across Observations\n(Sorted by Sepal Length)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Observation Index (Sorted)')
        ax.set_ylabel('Measurement (cm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print("✅ Line chart created: Feature trends across observations")
    
    def _create_bar_chart(self, ax):
        """Bar chart comparing average values across species"""
        # Calculate average petal length per species
        avg_by_species = self.df.groupby('species')['petal length (cm)'].mean().sort_values()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(avg_by_species.index, avg_by_species.values, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f} cm', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('📊 Average Petal Length by Species', fontsize=12, fontweight='bold')
        ax.set_xlabel('Species')
        ax.set_ylabel('Average Petal Length (cm)')
        ax.tick_params(axis='x', rotation=45)
        
        print("✅ Bar chart created: Average petal length by species")
    
    def _create_histogram(self, ax):
        """Histogram of sepal length distribution"""
        # Plot histogram with KDE
        sns.histplot(data=self.df, x='sepal length (cm)', hue='species', ax=ax, 
                    alpha=0.7, kde=True, element='step')
        
        ax.set_title('📋 Distribution of Sepal Length by Species', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sepal Length (cm)')
        ax.set_ylabel('Frequency')
        ax.legend(title='Species')
        
        print("✅ Histogram created: Distribution of sepal length")
    
    def _create_scatter_plot(self, ax):
        """Scatter plot of sepal length vs petal length"""
        scatter = sns.scatterplot(data=self.df, x='sepal length (cm)', y='petal length (cm)', 
                                 hue='species', style='species', s=100, ax=ax, alpha=0.8)
        
        ax.set_title('🔍 Sepal Length vs Petal Length', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sepal Length (cm)')
        ax.set_ylabel('Petal Length (cm)')
        ax.legend(title='Species')
        
        # Add correlation coefficient
        correlation = self.df['sepal length (cm)'].corr(self.df['petal length (cm)'])
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               fontweight='bold')
        
        print("✅ Scatter plot created: Sepal length vs petal length")
    
    def _create_advanced_visualizations(self):
        """Create additional advanced visualizations"""
        print("\n🎨 Creating Advanced Visualizations...")
        
        # 1. Pairplot for comprehensive relationships
        plt.figure(figsize=(12, 10))
        pairplot = sns.pairplot(self.df, hue='species', diag_kind='hist', palette='husl')
        pairplot.fig.suptitle('Iris Dataset - Pairwise Relationships', y=1.02, fontweight='bold')
        plt.show()
        
        # 2. Heatmap of correlations
        plt.figure(figsize=(10, 8))
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('🔥 Correlation Heatmap of Iris Features', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # 3. Boxplot for outlier detection
        plt.figure(figsize=(12, 6))
        melted_df = self.df.melt(id_vars=['species'], 
                                value_vars=['sepal length (cm)', 'sepal width (cm)', 
                                          'petal length (cm)', 'petal width (cm)'])
        
        sns.boxplot(data=melted_df, x='variable', y='value', hue='species')
        plt.title('📦 Distribution and Outliers by Species and Feature', fontweight='bold')
        plt.xlabel('Feature')
        plt.ylabel('Measurement (cm)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("✅ Advanced visualizations created!")
    
    def identify_patterns(self):
        """Identify and report interesting patterns"""
        if self.df is None:
            print("❌ No data loaded!")
            return
            
        print("\n" + "="*50)
        print("PATTERN IDENTIFICATION & INSIGHTS")
        print("="*50)
        
        # Calculate correlations
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlations = numeric_df.corr()
        
        print("\n🔍 Key Findings:")
        print("1. Species Differentiation:")
        print("   - Setosa has significantly smaller petals compared to other species")
        print("   - Virginica generally has the largest measurements")
        print("   - Versicolor falls between Setosa and Virginica")
        
        print("\n2. Strong Correlations:")
        strong_corr = correlations.unstack().sort_values(ascending=False)
        strong_corr = strong_corr[strong_corr < 1.0]  # Remove self-correlations
        print(f"   - Highest correlation: {strong_corr.index[0]} = {strong_corr.iloc[0]:.3f}")
        
        print("\n3. Measurement Patterns:")
        species_stats = self.df.groupby('species').mean()
        for species in species_stats.index:
            max_feature = species_stats.loc[species].idxmax()
            min_feature = species_stats.loc[species].idxmin()
            print(f"   - {species}: Largest {max_feature}, Smallest {min_feature}")
        
        print("\n4. Data Quality:")
        print("   - No missing values detected")
        print("   - All measurements are in consistent units (cm)")
        print("   - Balanced dataset with 50 samples per species")

def main():
    """Main function to run the complete analysis"""
    print("🚀 Starting Comprehensive Data Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = DataAnalyzer()
    
    try:
        # Load data
        if not analyzer.load_iris_dataset():
            return
        
        # Perform analysis
        analyzer.basic_analysis()
        analyzer.group_analysis()
        analyzer.identify_patterns()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()