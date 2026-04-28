import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processor import Country_Eda

st.set_page_config(page_title="COP32 Climate Dashboard", layout="wide")

@st.cache_data
def get_combined_data(countries):
    """
    Fetches data from Google Drive URLs stored in secrets.toml,
    then uses Country_Eda for processing.
    """
    combined_list = []
    for name in countries:
       
        eda = Country_Eda(name.lower())
        
        secret_key = f"{name.upper()}_DATA"
        
        try:
            data_url = st.secrets[secret_key]
           
            eda.df = pd.read_csv(data_url)
            
            
            eda.specific_country()
            eda.date_parser()
            eda.check_outliers() 
            combined_list.append(eda.df)
            
        except KeyError:
            st.error(f"Secret key '{secret_key}' not found in Streamlit Secrets.")
            continue
        except Exception as e:
            st.error(f"Error loading {name}: {e}")
            continue
            
    return pd.concat(combined_list, ignore_index=True)

def main():
    st.title("📊 Interactive Climate Analysis: Regional Comparison")
    st.markdown("Use the sidebar to filter data and compare country performance for **COP32** planning.")

    
    st.sidebar.header("Country Selection")
    available_countries = ["Ethiopia", "Sudan", "Nigeria", "Tanzania", "Kenya"]
    selected_countries = st.sidebar.multiselect(
        "Compare Countries", 
        options=available_countries, 
        default=["Ethiopia", "Sudan"]
    )

    year_range = st.sidebar.slider(
        "Year Range", 
        2015, 2026, (2015, 2026)
    )

    variable = st.sidebar.selectbox(
        "Climate Variable", 
        options=['T2M', 'PRECTOTCORR', 'RH2M', 'T2M_MAX', 'T2M_MIN']
    )

    if not selected_countries:
        st.warning("Please select at least one country to begin.")
        return

   
    full_df = get_combined_data(selected_countries)
    
   
    filtered_df = full_df[
        (full_df['DATE'].dt.year >= year_range[0]) & 
        (full_df['DATE'].dt.year <= year_range[1])
    ]

    
    tab1, tab2 = st.tabs(["📈 Temporal Trends", "📊 Distribution & Volatility"])

    with tab1:
        st.subheader(f"Interactive {variable} Trend")
       
        monthly_comp = filtered_df.groupby(['COUNTRY', pd.Grouper(key='DATE', freq='MS')])[variable].mean().reset_index()
        
        fig_trend = px.line(
            monthly_comp, 
            x='DATE', 
            y=variable, 
            color='COUNTRY',
            labels={'DATE': 'Year', variable: f'{variable} (°C/mm)'},
            template="plotly_white",
            hover_name="COUNTRY"
        )
        fig_trend.update_layout(hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Precipitation Distribution")
            fig_box = px.box(
                filtered_df, 
                x='COUNTRY', 
                y='PRECTOTCORR', 
                color='COUNTRY',
                points="outliers", 
                title="Rainfall Volatility & Outliers",
                template="plotly_white"
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with col2:
            st.subheader("Relative Humidity vs Temperature")
            fig_scatter = px.scatter(
                filtered_df, 
                x='T2M', 
                y='RH2M', 
                color='COUNTRY', 
                size='PRECTOTCORR',
                opacity=0.5,
                title="Heat-Moisture Relationship (Bubble size = Rainfall)",
                template="plotly_white"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

   
    st.divider()
    st.subheader("Comparative Summary Statistics")
    summary = filtered_df.groupby('COUNTRY')[variable].agg(['mean', 'std', 'min', 'max']).rename(
        columns={'mean': 'Average', 'std': 'Volatility (Std)', 'min': 'Minimum', 'max': 'Maximum'}
    )
    st.table(summary)

if __name__ == "__main__":
    main()