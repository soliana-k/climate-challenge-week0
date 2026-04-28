import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging

from scipy import stats
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class Country_Eda:
    def __init__(self, name):
       self.name=name
       self.df=None
        
    def load_data(self):
        """Loads data with error handling for missing files."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, '..', 'csv', f'{self.name}.csv')

            file_path = os.path.normpath(file_path)
    
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Could not find the file at: {file_path}")

            self.df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data for {self.name}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise



    def specific_country(self):
        self.df['COUNTRY']=self.name.capitalize()
        logger.info('Specific country set')

    def date_parser(self):
        """Parses dates with validation."""
        try:
            if 'YEAR' not in self.df.columns or 'DOY' not in self.df.columns:
                raise KeyError("Missing YEAR or DOY columns for date parsing.")
            
            self.df['DATE']=pd.to_datetime(
            self.df['YEAR'].astype(str) + ' ' + self.df['DOY'].astype(str), format='%Y %j')

            self.df['MONTH']=self.df['DATE'].dt.month

            logger.info('date parsed successfully & month extracted succesfully')
        except Exception as e:
            logger.error(f"Error in date_parser: {e}")
            raise
            
    
    def data_quality(self):
        """Checks quality and handles nulls systematically."""
        print(f'--------------Data Quality Report for {self.name.capitalize()}-----------')
        print("\n--- Data Information ---\n")
        print(self.df.info())
        
        dup_count=self.df.duplicated().sum()
        if dup_count > 0:
            self.df=self.df.drop_duplicates()
            logger.info(f'Found and dropped {dup_count} duplicates')
        else:
            logger.info('Duplicates: no duplicates found')
        self.df=self.df.replace(-999, np.nan)
        logger.info('Replaced all -999 occurrences with NaN.')
        
        print('\n-------------- missing data check ---------------\n')
       
        null_counts = self.df.isna().sum()
        null_perc = (null_counts / len(self.df)) * 100
        
        print(null_perc[null_perc > 0] if null_perc.sum() > 0 else "no missing values detected.")

        too_nulls = null_perc[null_perc > 5]
        if not too_nulls.empty:
            for col, val in too_nulls.items():
                logger.warning(f"aolumn '{col}' has {val:.2f}% missing data. this may affect seasonal analysis.")
        else:
            logger.info("all columns have < 5% missing data. good for analysis.")
        
        logger.info("\n--- Summary Statistics ---\n")
        stats_summary = self.df.describe()
        print(stats_summary) 

        print('-------------------------------------------------\n\n')
   

    def check_outliers(self, threshold=3):
        """ Checks for outliers and reasons for decisions. """
        cols=['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'WS2M', 'WS2M_MAX']
        z_score=self.df[cols].apply(stats.zscore)
        norm_z_score=np.abs(z_score)

        outlier_indices=(norm_z_score>threshold).any(axis=1)
        outliers=self.df[outlier_indices]
        perc=(len(outliers)/len(self.df)) * 100
        
        logger.info('------------ Outliers Report-------------------\n')
        if perc < 1:
            reasoning = "Clean Data: Minimal outliers detected. Retaining all points."
        elif perc < 7:
            reasoning = f"Natural Variability: {perc:.2f}% outliers is expected in climate data representing extreme weather events (storms/heatwaves). Decision: RETAIN for analysis."
        else:
            reasoning = "High Volatility: Significant outliers detected. Suggests sensor noise or extreme climate shifts. Decision: INSPECT/CAP."

        logger.warning(f'total rows flagged as outliers {len(outliers)} and the percentage is {perc}%\n')  

        logger.info(f"REASONING: {reasoning}\n")
        print('------------- Missing Value Handling---------------')
        initial=len(self.df)
        min_threshold = int(len(self.df.columns) * 0.7)
        self.df=self.df.dropna(thresh=min_threshold)
        dropped=initial - len(self.df)
        if dropped > 0:
            logger.warning(f"dropped {dropped} rows because they were more than 30% empty.")
        
        self.df[cols] = self.df[cols].ffill()
        logger.info("applied ffill to remaining missing weather variables.")
        print('-------------------------------------------------\n')



    def plotting_average_monthly_T2M(self):
        """ Plot average monthly T2M"""
        logger.info('________Preparing resampled Monthly data for T2M________\n')
        
    
        monthly_df = self.df.set_index('DATE')['T2M'].resample('MS').mean().reset_index()
        
        
        warmest = monthly_df.loc[monthly_df['T2M'].idxmax()]
        coolest = monthly_df.loc[monthly_df['T2M'].idxmin()]

        
        x_coords = np.arange(len(monthly_df))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_coords, monthly_df['T2M'])
        
        
        temp_range = monthly_df['T2M'].max() - monthly_df['T2M'].min()
        recent_avg = monthly_df['T2M'].tail(12).mean()
        historic_avg = monthly_df['T2M'].head(12).mean()

        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=monthly_df, x='DATE', y='T2M', ax=ax, color='#2c7fb8', linewidth=2.5, label='Monthly Mean')
        
        
        ax.plot(monthly_df['DATE'], intercept + slope * x_coords, color='orange', linestyle='--', label='Overall Trend')

        
        ax.annotate(f'Warmest: {warmest["T2M"]:.1f}°C', xy=(warmest['DATE'], warmest['T2M']), 
                    xytext=(0, 20), textcoords='offset points', ha='center', color='red', weight='bold',
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.annotate(f'Coolest: {coolest["T2M"]:.1f}°C', xy=(coolest['DATE'], coolest['T2M']), 
                    xytext=(0, -15), textcoords='offset points', ha='center', color='blue', weight='bold',
                    arrowprops=dict(arrowstyle='->', color='blue'))

        ax.xaxis.set_major_locator(mdates.YearLocator()) 
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.title(f'Temperature Trends - {self.name.capitalize()} (2015–2026)\n', fontsize=14)
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.show()

       
        logger.info(f"\n--- Automated Insights for {self.name.capitalize()} ---")
        
        
        trend_desc = "increasing (warming)" if slope > 0 else "decreasing (cooling)"
        logger.info(f"1. TREND: The temperature is generally {trend_desc} at a rate of {slope*12:.3f}°C per year.")
        
        
        logger.info(f"2. VOLATILITY: The total temperature swing across the period is {temp_range:.1f}°C.")
        if temp_range > 15:
            logger.info("   Note: High seasonal variance detected (likely a high-altitude or inland climate).")
        
        
        if abs(recent_avg - historic_avg) > 1.0:
            status = "warmer" if recent_avg > historic_avg else "cooler"
            logger.info(f"3. ANOMALY: The most recent 12 months are {abs(recent_avg - historic_avg):.2f}°C {status} than the first 12 months.")
        else:
            logger.info("3. ANOMALY: No significant long-term shift detected between early and late periods.")
        print("---------------------------------------------------\n")



    def monthly_total_precipitation(self):
        """ Plot monthly total precipitation."""
        
        monthly=self.df.groupby('MONTH')['PRECTOTCORR'].sum().reset_index()
        month_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        monthly['MONTH_NAME'] = monthly['MONTH'].map(month_map)

        peak_rainy_row=monthly['PRECTOTCORR'].idxmax()
        peak_value = monthly.loc[peak_rainy_row, 'PRECTOTCORR']
        peak_month = monthly.loc[peak_rainy_row, 'MONTH_NAME']
        
        fig, ax= plt.subplots(figsize=(14,7))
        sns.barplot(data=monthly, x='MONTH_NAME', y='PRECTOTCORR', ax=ax, palette='Blues_d')
        ax.annotate(f'Peak: {peak_month}\n{peak_value:.1f} mm', 
                    xy=(peak_rainy_row, peak_value), 
                    xytext=(0, 30), 
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontweight='bold', color='black',
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

        plt.title(f'Total Monthly Precipitation - {self.name.capitalize()}')
        plt.ylabel('Total Precipitation (mm)/day')
        plt.ylim(0, peak_value * 1.2)
        plt.show()
        

    def correlation_and_relationship(self):
        """Checks and plots correlations and relationships"""
        fig, ax= plt.subplots(figsize=(10,6))
        corr=self.df.corr(numeric_only=True)
        sns.heatmap(data=corr, annot=True, cmap='coolwarm', ax=ax)
        plt.show()
        corr_m=corr.unstack()
        sorted_corr=corr_m[corr_m < 1].sort_values(ascending=False, key=abs)
        top_3 = sorted_corr.head(6)[::2]
        logger.info(f"\n--- Top 3 Strongest Relationships for {self.name.capitalize()} ---")
        for i, ((var1, var2), val) in enumerate(top_3.items(), 1):
            strength = "Strong" if abs(val) > 0.7 else "Moderate"
            direction = "Positive" if val > 0 else "Negative"
            logger.info(f"{i}. {var1} & {var2}: {val:.2f} ({strength} {direction})")
        print("-----------------------------------------------------------\n")


        print('_____scatter plot_____\n\n')
        fig, ax= plt.subplots(figsize=(10,6))
        sns.scatterplot(data=self.df, x='T2M', y='RH2M', ax=ax)
        plt.title('Mean daily air temperature at 2 meters vs Relative humidity at 2 meters ')
        plt.show()
        print('_____scatter plot 2_____\n\n')
        fig, ax= plt.subplots(figsize=(10,6))
        sns.scatterplot(data=self.df, x='T2M_RANGE', y='WS2M', ax=ax)
        plt.title('Daily temperature range vs Mean daily wind speed at 2 meters')
        plt.show()
    

    def distribution_analysis(self):
        """ check and analyse the distribution of PRECTOTCORR"""
        skew=self.df['PRECTOTCORR'].skew()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sns.histplot(self.df['PRECTOTCORR'], kde=True, ax=ax1, color='blue')
        ax1.set_title(f'Original Distribution (Skew: {skew:.2f})')
        
        if skew > 1:
            logger.info('The distribution of total daily precipitation is strongly right-skewed. the shape would be long right tail, bulk on the left')
            log_data = np.log1p(self.df['PRECTOTCORR'])
            
            sns.histplot(log_data, kde=True, ax=ax2, color='green')
            ax2.set_title('Log-Transformed Distribution (Balanced)')
            ax2.set_xlabel('Log(Precipitation + 1)')
        elif skew < -1:
            logger.info('The distribution of total daily precipitation is strongly left-skewed. the shape would be bulk on the right, tail to the left')
            ax2.axis('off')
        elif -0.5 < skew < 0.5:
            logger.info('the distribution of total daily precipitation is fairly symmetric, Bell curve')
            ax2.axis('off')
        else:
            logger.info('The distribution of total daily precipitation is symmetric and normal. the shape would be bell-shaped, tails similar')
            ax2.axis('off')
        
        # bubble
        fig, ax=plt.subplots(figsize=(10,6))
        sns.scatterplot(
            data=self.df, 
            x='T2M', 
            y='RH2M', 
            size='PRECTOTCORR', 
            hue= 'PRECTOTCORR', 
            palette='viridis',
            sizes=(20, 500),  
            alpha=0.7,        
            edgecolor='black'
        )
        plt.show()

    def export_to_csv(self):
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    
        target_dir = os.path.join(parent_dir, 'data')
        
        file_path = os.path.join(target_dir, f'{self.name}_clean.csv')
        self.df.to_csv(file_path, index=False, encoding='utf-8')
        logger.info('______exported successfully_______')



       


    def run(self):
        self.load_data()
        self.specific_country()
        self.date_parser()
        self.data_quality()
        self.check_outliers()
        self.plotting_average_monthly_T2M()
        self.monthly_total_precipitation()
        self.correlation_and_relationship()
        self.distribution_analysis()
        self.export_to_csv()

        
    
    