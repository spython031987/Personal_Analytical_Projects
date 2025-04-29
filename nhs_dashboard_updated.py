import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import io

# Set page config
st.set_page_config(
    page_title="NHS Performance Dashboard",
    page_icon="🏥",
    layout="wide",
)

# Function to load and process data
def load_data(uploaded_file=None):
    if uploaded_file is None:
        st.error("Please upload an Excel file to continue.")
        st.stop()
    
    # Load Excel data from the uploaded file
    try:
        df = pd.read_excel(uploaded_file)
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Rename columns for easier access
        column_mapping = {
            'Organization Code ': 'code',
            'NHS Trust Name': 'name',
            'Region': 'region',
            'Percentage of attendance within 4 hours or less (all)': 'percentageIn4Hours',
            'Total Emergency Admissions': 'totalEmergencyAdmissions',
            'Number of patients spending >12 hours from decision to admit to admission': 'waits12HoursPlus',
            'Customer Satisfaction\nTotal Responses': 'customerSatisfactionTotal',
            'Customer Satisfaction\nPercentage of Positive Responses': 'customerSatisfactionPositive',
            'Customer Satisfaction\nPercentage of Negative Responses': 'customerSatisfactionNegative',
            'Referral to Treatment Waiting Times\nAverage (median) waiting time (in weeks)': 'waitingTimeWeeks',
            'Total Head Count': 'totalHeadCount',
            'G&A beds available': 'bedsAvailable',
            'Occupancy Rate': 'occupancyRate',
            '% of G&A beds occupied by patients for 21 or more days': 'longStayPercentage'
        }
        
        # Rename columns where they exist in the dataframe
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Fill NaN values with 0
        numeric_cols = ['totalEmergencyAdmissions', 'waits12HoursPlus', 'customerSatisfactionTotal',
                       'customerSatisfactionPositive', 'customerSatisfactionNegative', 'waitingTimeWeeks',
                       'totalHeadCount', 'bedsAvailable', 'occupancyRate', 'longStayPercentage']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Filter out rows with missing essential data
        df = df.dropna(subset=['name', 'region', 'percentageIn4Hours'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Format numbers with commas
def format_number(num):
    return f"{num:,.0f}"

# Format percentages
def format_percentage(value):
    if pd.isna(value):
        return "N/A"
    return f"{value*100:.1f}%"

# Function to create region data
def prepare_region_data(df):
    # Get all regions
    regions = ['All'] + sorted(df['region'].unique().tolist())
    
    # Calculate regional performance
    region_data = []
    
    for region in regions:
        if region == 'All':
            continue
            
        region_trusts = df[df['region'] == region]
        total_admissions = region_trusts['totalEmergencyAdmissions'].sum()
        avg_performance = region_trusts['percentageIn4Hours'].mean()
        total_12hr_waits = region_trusts['waits12HoursPlus'].sum()
        avg_waiting_time = region_trusts['waitingTimeWeeks'].mean()
        occupancy_rate = region_trusts['occupancyRate'].mean()
        
        # Create a shorter display name for the region
        display_name = region.replace('NHS England', '').strip()
        
        region_data.append({
            'name': display_name,
            'fullRegionName': region,
            'totalEmergencyAdmissions': total_admissions,
            'percentageIn4Hours': avg_performance,
            'total12HourWaits': total_12hr_waits,
            'avgWaitingTime': avg_waiting_time,
            'occupancyRate': occupancy_rate
        })
    
    # Sort by performance
    region_data = sorted(region_data, key=lambda x: x['percentageIn4Hours'], reverse=True)
    return region_data

# Function to calculate national statistics
def calculate_national_stats(df):
    stats = {
        'totalTrusts': len(df),
        'totalEmergencyAdmissions': df['totalEmergencyAdmissions'].sum(),
        'averagePercentageIn4Hours': df['percentageIn4Hours'].mean(),
        'total12HourWaits': df['waits12HoursPlus'].sum(),
        'averageOccupancyRate': df['occupancyRate'].mean(),
        'averageWaitingTime': df['waitingTimeWeeks'].mean(),
        'bestPerformer': df.loc[df['percentageIn4Hours'].idxmax()].to_dict() if len(df) > 0 else None,
        'worstPerformer': df.loc[df['percentageIn4Hours'].idxmin()].to_dict() if len(df) > 0 else None
    }
    return stats

# Function to prepare trust metrics for radar chart
def prepare_trust_metrics(df, selected_trust):
    if not selected_trust:
        return None
    
    trust = df[df['name'] == selected_trust].iloc[0]
    
    # Calculate national averages for comparison
    avg_percentageIn4Hours = df['percentageIn4Hours'].mean()
    avg_occupancyRate = df['occupancyRate'].mean()
    avg_waitingTimeWeeks = df['waitingTimeWeeks'].mean()
    avg_customerSatisfaction = df['customerSatisfactionPositive'].mean()
    avg_longStayPercentage = df['longStayPercentage'].mean()
    
    metrics = [
        {
            'subject': '4-Hour Target',
            'trustValue': trust['percentageIn4Hours'],
            'nationalAvg': avg_percentageIn4Hours,
            'fullMark': 1,
            # Higher is better for this metric
            'trustScore': trust['percentageIn4Hours'],
            'nationalScore': avg_percentageIn4Hours
        },
        {
            'subject': 'Bed Occupancy',
            'trustValue': trust['occupancyRate'],
            'nationalAvg': avg_occupancyRate,
            'fullMark': 1,
            # Lower is better for this metric, so invert the score
            'trustScore': 1 - trust['occupancyRate'],
            'nationalScore': 1 - avg_occupancyRate
        },
        {
            'subject': 'Waiting Time',
            'trustValue': trust['waitingTimeWeeks'],
            'nationalAvg': avg_waitingTimeWeeks,
            'fullMark': 20,
            # Lower is better, normalize to 0-1 where 1 is best
            'trustScore': 1 - (trust['waitingTimeWeeks'] / 20),  # Assuming 20 weeks is maximum
            'nationalScore': 1 - (avg_waitingTimeWeeks / 20)
        },
        {
            'subject': 'Patient Satisfaction',
            'trustValue': trust['customerSatisfactionPositive'],
            'nationalAvg': avg_customerSatisfaction,
            'fullMark': 1,
            # Higher is better
            'trustScore': trust['customerSatisfactionPositive'],
            'nationalScore': avg_customerSatisfaction
        },
        {
            'subject': 'Long-stay %',
            'trustValue': trust['longStayPercentage'],
            'nationalAvg': avg_longStayPercentage,
            'fullMark': 0.5,
            # Lower is better, normalize to 0-1 where 1 is best
            'trustScore': 1 - (trust['longStayPercentage'] / 0.5),  # Assuming 50% is maximum
            'nationalScore': 1 - (avg_longStayPercentage / 0.5)
        }
    ]
    
    return metrics

# Main function
def main():
    # Title and date
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("NHS Performance Dashboard")
    with col2:
        st.write(f"Data for March 2025")
    
    # File upload section
    st.write("### Upload NHS Data")
    uploaded_file = st.file_uploader("Upload Excel file containing NHS performance data", 
                                     type=["xlsx", "xls"],
                                     help="The file should contain NHS Trust data with standard column names")
    
    # Sample file download option
    st.markdown("Need a sample file? [Download template](https://example.com/NHS_Viz_Data_Sample.xlsx)")
    
    # Only proceed if a file has been uploaded
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading NHS data..."):
            df = load_data(uploaded_file)
            
        # Calculate national statistics and region data
        national_stats = calculate_national_stats(df)
        region_data = prepare_region_data(df)
        
        # Display file information
        st.success(f"File loaded successfully: {uploaded_file.name}")
        st.write(f"Total NHS Trusts: {national_stats['totalTrusts']}")
        
        # Key metrics section
        st.subheader("Key Metrics")
        key_metrics_cols = st.columns(4)
        
        with key_metrics_cols[0]:
            st.metric(
                label="Emergency Admissions",
                value=format_number(national_stats['totalEmergencyAdmissions']),
                help=f"Across {national_stats['totalTrusts']} NHS Trusts"
            )
            
        with key_metrics_cols[1]:
            st.metric(
                label="4-Hour Performance",
                value=format_percentage(national_stats['averagePercentageIn4Hours']),
                help="National Average (Target: 95%)"
            )
            
        with key_metrics_cols[2]:
            st.metric(
                label="12hr+ Waits",
                value=format_number(national_stats['total12HourWaits']),
                help="Decision to admit to admission"
            )
            
        with key_metrics_cols[3]:
            st.metric(
                label="Bed Occupancy",
                value=format_percentage(national_stats['averageOccupancyRate']),
                help="National Average (Target: <85%)"
            )
        
        # Best and worst performers
        st.subheader("Best and Worst Performers")
        best_worst_cols = st.columns(2)
        
        with best_worst_cols[0]:
            best_name = national_stats['bestPerformer']['name'] if national_stats['bestPerformer'] else 'N/A'
            best_perf = national_stats['bestPerformer']['percentageIn4Hours'] if national_stats['bestPerformer'] else 0
            
            st.success(f"**Best Performer: {best_name}**")
            progress_best = st.progress(float(best_perf))
            st.write(f"4-Hour Target Performance: {format_percentage(best_perf)}")
        
        with best_worst_cols[1]:
            worst_name = national_stats['worstPerformer']['name'] if national_stats['worstPerformer'] else 'N/A'
            worst_perf = national_stats['worstPerformer']['percentageIn4Hours'] if national_stats['worstPerformer'] else 0
            
            st.error(f"**Worst Performer: {worst_name}**")
            progress_worst = st.progress(float(worst_perf))
            st.write(f"4-Hour Target Performance: {format_percentage(worst_perf)}")
        
        # Regional performance chart
        st.subheader("Regional 4-Hour Performance")
        
        # Convert region data to DataFrame for plotting
        region_df = pd.DataFrame(region_data)
        
        # Create a horizontal bar chart
        fig_region = px.bar(
            region_df,
            y='name',
            x='percentageIn4Hours',
            orientation='h',
            labels={'percentageIn4Hours': '4-Hour Performance', 'name': 'Region'},
            title='Regional 4-Hour Performance',
            height=500
        )
        
        # Set colors based on performance
        fig_region.update_traces(
            marker_color=[
                '#00C49F' if x >= 0.85 else '#FF8042' if x >= 0.70 else '#FF6464' 
                for x in region_df['percentageIn4Hours']
            ]
        )
        
        # Format x-axis as percentage
        fig_region.update_layout(
            xaxis_tickformat='.0%',
            xaxis_title="4-Hour Performance",
            yaxis_title="Region",
            xaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig_region, use_container_width=True)
        
        # Top & Bottom Performers section
        st.subheader("Top & Bottom Performers")
        
        top_bottom_cols = st.columns(2)
        
        # Get top performers (filter out very small providers)
        top_performers = df[df['totalEmergencyAdmissions'] > 1000].sort_values(
            by='percentageIn4Hours', ascending=False).head(5)
        
        # Get bottom performers (filter out very small providers)
        bottom_performers = df[df['totalEmergencyAdmissions'] > 1000].sort_values(
            by='percentageIn4Hours', ascending=True).head(5)
        
        with top_bottom_cols[0]:
            st.write("##### Top 5 Performers (4-Hour Target)")
            
            for _, trust in top_performers.iterrows():
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.write(trust['name'])
                with col2:
                    st.progress(float(trust['percentageIn4Hours']))
                    st.write(format_percentage(trust['percentageIn4Hours']))
        
        with top_bottom_cols[1]:
            st.write("##### Bottom 5 Performers (4-Hour Target)")
            
            for _, trust in bottom_performers.iterrows():
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.write(trust['name'])
                with col2:
                    st.progress(float(trust['percentageIn4Hours']))
                    st.write(format_percentage(trust['percentageIn4Hours']))
        
        # Trust Performance section
        st.subheader("Trust Performance")
        
        # Sidebar filters
        filter_cols = st.columns([1, 2, 2])
        
        with filter_cols[0]:
            # Sort options
            sort_options = {
                'percentageIn4Hours': '4-Hour Performance',
                'totalEmergencyAdmissions': 'Emergency Admissions',
                'waits12HoursPlus': '12hr+ Waits',
                'occupancyRate': 'Bed Occupancy'
            }
            sort_by = st.selectbox("Sort by", options=list(sort_options.keys()), 
                                 format_func=lambda x: sort_options[x],
                                 index=0)
            
            sort_direction = st.radio("Sort direction", ["Descending", "Ascending"], 
                                     index=0, horizontal=True)
            
        with filter_cols[1]:
            # Region filter
            regions = ['All'] + sorted(df['region'].unique().tolist())
            selected_region = st.selectbox("Filter by region", options=regions)
        
        with filter_cols[2]:
            # Search filter
            search_term = st.text_input("Search trusts", value="")
        
        # Filter and sort data
        filtered_df = df.copy()
        
        # Apply region filter
        if selected_region != 'All':
            filtered_df = filtered_df[filtered_df['region'] == selected_region]
        
        # Apply search filter
        if search_term:
            filtered_df = filtered_df[filtered_df['name'].str.contains(search_term, case=False)]
        
        # Sort data
        ascending = True if sort_direction == "Ascending" else False
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
        
        # Calculate aggregates for the filtered data
        filtered_stats = {
            'totalTrusts': len(filtered_df),
            'totalEmergencyAdmissions': filtered_df['totalEmergencyAdmissions'].sum(),
            'totalWaits12HoursPlus': filtered_df['waits12HoursPlus'].sum(),
            'averagePercentageIn4Hours': filtered_df['percentageIn4Hours'].mean() if len(filtered_df) > 0 else 0,
            'averageOccupancyRate': filtered_df['occupancyRate'].mean() if len(filtered_df) > 0 else 0
        }
        
        # Trust dropdown for drill-down
        trust_options = [''] + sorted(filtered_df['name'].tolist())
        selected_trust = st.selectbox("Select Trust for Detail View", options=trust_options, index=0)
        
        # Display trust details if selected
        if selected_trust:
            trust_metrics = prepare_trust_metrics(df, selected_trust)
            
            st.write(f"### Detail View: {selected_trust}")
            
            # Create a radar chart for the trust
            categories = [metric['subject'] for metric in trust_metrics]
            
            fig = go.Figure()
            
            # Add traces for trust and national average
            fig.add_trace(go.Scatterpolar(
                r=[metric['trustScore'] for metric in trust_metrics],
                theta=categories,
                fill='toself',
                name=selected_trust
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=[metric['nationalScore'] for metric in trust_metrics],
                theta=categories,
                fill='toself',
                name='National Average'
            ))
            
            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                height=500
            )
            
            radar_metrics_cols = st.columns([2, 1])
            
            with radar_metrics_cols[0]:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show trust metrics table
            with radar_metrics_cols[1]:
                st.write("#### Key Metrics")
                
                for metric in trust_metrics:
                    st.write(f"**{metric['subject']}**")
                    
                    if metric['subject'] == 'Waiting Time':
                        trust_value = f"{metric['trustValue']:.1f} weeks"
                        national_value = f"{metric['nationalAvg']:.1f} weeks"
                    else:
                        trust_value = format_percentage(metric['trustValue'])
                        national_value = format_percentage(metric['nationalAvg'])
                    
                    trust_cols = st.columns(2)
                    with trust_cols[0]:
                        st.write(f"Trust: {trust_value}")
                    with trust_cols[1]:
                        st.write(f"National: {national_value}")
                    
                    # Comparison text
                    if (metric['subject'] == 'Waiting Time' or 
                        metric['subject'] == 'Bed Occupancy' or 
                        metric['subject'] == 'Long-stay %'):
                        # Lower is better
                        if metric['trustValue'] < metric['nationalAvg']:
                            st.success("Better than average")
                        else:
                            st.error("Worse than average")
                    else:
                        # Higher is better
                        if metric['trustValue'] > metric['nationalAvg']:
                            st.success("Better than average")
                        else:
                            st.error("Worse than average")
                    
                    st.write("---")
        
        # Show filtered stats and data table if no trust selected
        else:
            # Summary stats for filtered data
            filtered_stats_cols = st.columns(4)
            
            with filtered_stats_cols[0]:
                st.metric("Trusts", filtered_stats['totalTrusts'])
                
            with filtered_stats_cols[1]:
                st.metric("Emergency Admissions", format_number(filtered_stats['totalEmergencyAdmissions']))
                
            with filtered_stats_cols[2]:
                st.metric("Avg 4-Hour Performance", format_percentage(filtered_stats['averagePercentageIn4Hours']))
                
            with filtered_stats_cols[3]:
                st.metric("12hr+ Waits", format_number(filtered_stats['totalWaits12HoursPlus']))
            
            # Data table
            st.write("#### Trust Data")
            
            # Format the dataframe for display
            display_df = filtered_df[['name', 'totalEmergencyAdmissions', 'percentageIn4Hours', 
                                    'waits12HoursPlus', 'occupancyRate']].copy()
            
            # Format columns
            display_df['totalEmergencyAdmissions'] = display_df['totalEmergencyAdmissions'].apply(
                lambda x: format_number(x) if pd.notna(x) else 'N/A')
            display_df['percentageIn4Hours'] = display_df['percentageIn4Hours'].apply(
                lambda x: format_percentage(x) if pd.notna(x) else 'N/A')
            display_df['waits12HoursPlus'] = display_df['waits12HoursPlus'].apply(
                lambda x: format_number(x) if pd.notna(x) else 'N/A')
            display_df['occupancyRate'] = display_df['occupancyRate'].apply(
                lambda x: format_percentage(x) if pd.notna(x) else 'N/A')
            
            # Rename columns for display
            display_df.columns = ['Trust', 'Emergency Admissions', '4-Hour Performance', '12hr+ Waits', 'Bed Occupancy']
            
            # Show the dataframe
            st.dataframe(display_df, use_container_width=True)
        
        # Download processed data option
        st.subheader("Download Data")
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="Download processed data as CSV",
            data=csv_data,
            file_name=f"nhs_data_processed_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download the processed data with standardized column names"
        )
        
        # Footer
        st.caption("Data source: NHS Digital, Emergency Care Dataset, March 2025")
    else:
        # Show welcome message and instructions if no file is uploaded
        st.info("👆 Please upload an Excel file to see the NHS Performance Dashboard")
        
        # Placeholder image or description of what the dashboard will look like
        st.write("### Dashboard Preview")
        st.write("""
        Once you upload an Excel file with NHS performance data, you'll be able to:
        
        * View key national performance metrics
        * Compare regional performance
        * Explore individual trust performance
        * Filter and sort data by various criteria
        * Download processed data for further analysis
        
        The dashboard expects data with standard NHS column names including trust names, 
        regions, emergency admission volumes, 4-hour performance, and other key metrics.
        """)

if __name__ == "__main__":
    main()
