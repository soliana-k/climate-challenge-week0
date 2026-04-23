import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy import stats

class Country_Eda:
    def __init__(self, name):
       self.name=name
       self.df=None
        
    def load_data(self):
        self.df=pd.read_csv(f'../csv/{self.name}.csv')
        print('_____ Data Loaded Successfully________')



    def specific_country(self):
        self.df['COUNTRY']=self.name.capitalize()
        print('_________Specific country set________')

    def date_parser(self):
        self.df['Date']=pd.to_datetime(
        self.df['YEAR'].astype(str) + ' ' + self.df['DOY'].astype(str), 
        format='%Y %j')
        self.df['MONTH']=self.df['Date'].dt.month
        print('________date parsed successfully & month extracted succesfully______')
    
    def data_quality(self):
        self.df=self.df.drop_duplicates()
        self.df=self.df.replace(-999, np.nan)
        print(f'The amount of missing data is {self.df.isna().mean() * 100}%\n')
    
    def distributions(self):
        numerical_cols=self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            plt.figure(figsize=(10, 6))

            sns.histplot(data=self.df, x=col, kde=True)
            plt.title(f'Distribution of {col} in {self.name.capitalize()} Climate')
            plt.show()


    def check_outliers(self, threshold=3):
        cols=['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'WS2M', 'WS2M_MAX']
        z_score=self.df[cols].apply(stats.zscore)
        norm_z_score=np.abs(z_score)

        outlier_indices=(norm_z_score>threshold).any(axis=1)
        outliers=self.df[outlier_indices]
        print(f'total rows flagged as outliers {len(outliers)}')

    def plotting_average_monthly_T2M(self):
        print('________Preparing resampled Monthly data for T2M________')
        
        monthly_df = self.df.set_index('Date')['T2M'].resample('MS').mean().reset_index()

        warmest=monthly_df.loc[monthly_df['T2M'].idxmax()]
        coolest=monthly_df.loc[monthly_df['T2M'].idxmin()]


        fig, ax = plt.subplots(figsize=(10, 6))
        print('_______Plotting_______')
    
        sns.lineplot(data=monthly_df, x='Date', y='T2M', ax=ax, color='#2c7fb8', linewidth=2.5)

    
        ax.annotate(f'Warmest Month: {warmest["Date"].strftime("%b %Y")}\n{warmest["T2M"]:.1f}°C',
                xy=(warmest['Date'], warmest['T2M']),
                xytext=(0, 25), textcoords='offset points',
                ha='center', fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        ax.annotate(f'Coolest Month: {coolest["Date"].strftime("%b %Y")}\n{coolest["T2M"]:.1f}°C',
                xy=(coolest['Date'], coolest['T2M']),
                xytext=(0, -40), textcoords='offset points',
                ha='center', fontweight='bold', color='blue',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))


        ax.xaxis.set_major_locator(mdates.YearLocator()) 
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        plt.title('Monthly Average T2M (2015–2026)\n\n', fontsize=14, pad=20)
        plt.ylabel('Temperature (°C)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print('____Plotted successfully____')
        print(f'The coolest month is - {coolest }\n The warmest month is - {warmest}')

    def monthly_total_precipitation(self):
        
        monthly=self.df.groupby('MONTH')['PRECTOTCORR'].sum().reset_index()
        month_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        monthly['Month_Name'] = monthly['MONTH'].map(month_map)

        peak_rainy_row=monthly['PRECTOTCORR'].idxmax()
        peak_value = monthly.loc[peak_rainy_row, 'PRECTOTCORR']
        peak_month = monthly.loc[peak_rainy_row, 'Month_Name']
        
        fig, ax= plt.subplots(figsize=(14,7))
        sns.barplot(data=monthly, x='Month_Name', y='PRECTOTCORR', ax=ax, palette='Blues_d')
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
        fig, ax= plt.subplots(figsize=(10,6))
        sns.heatmap(data=self.df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
        plt.show()
        print('_____scatter plot_____')
        fig, ax= plt.subplots(figsize=(10,6))
        sns.scatterplot(data=self.df, x='T2M', y='RH2M', ax=ax)
        plt.title('Mean daily air temperature at 2 meters vs Relative humidity at 2 meters ')
        plt.show()
        print('_____scatter plot 2_____')
        fig, ax= plt.subplots(figsize=(10,6))
        sns.scatterplot(data=self.df, x='T2M_RANGE', y='WS2M', ax=ax)
        plt.title('Daily temperature range vs Mean daily wind speed at 2 meters')
        plt.show()
    

    def distribution_analysis(self):
        fig, ax=plt.subplots(figsize=(10,6))
        sns.histplot(data=self.df['PRECTOTCORR'], ax=ax, kde=True)
        plt.title('The distribution of total daily precipitation')
        plt.show()
        skew=self.df['PRECTOTCORR'].skew()
        if skew > 1:
            print('The distribution of total daily precipitation is strongly right-skewed. the shape would be long right tail, bulk on the left')
        elif skew < -1:
            print('The distribution of total daily precipitation is strongly left-skewed. the shape would be bulk on the right, tail to the left')
        elif -0.5 < skew < 0.5:
            print('the distribution of total daily precipitation is fairly symmetric, Bell curve')
        else:
            print('The distribution of total daily precipitation is symmetric and normal. the shape would be bell-shaped, tails similar')

        # bubble
        fig, ax=plt.subplots(figsize=(10,6))
        sns.scatterplot(
            data=self.df, 
            x='T2M', 
            y='RH2M', 
            size='PRECTOTCORR',   
            palette='viridis',
            sizes=(0, 80),  
            alpha=0.7,        
            edgecolor='black'
        )
        plt.show()



       


    def run(self):
        self.load_data()
        self.specific_country()
        self.date_parser()
        self.data_quality()
        #self.distributions()
        self.check_outliers()
        print(f'total missing values {self.df.isna().sum()}')
        self.plotting_average_monthly_T2M()
        self.monthly_total_precipitation()
        self.correlation_and_relationship()
        self.distribution_analysis()

        
    
    