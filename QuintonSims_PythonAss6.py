import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import random

# Checks if all required columns are in the DataFrame.
def has_cols(df, cols):
    return set(cols).issubset(df.columns)

#Prevents errors while dividing
def safe_div(numer, denom):
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    return numer / denom.replace({0: np.nan})

# Define the full state names for use in Tab 2
code_to_state = {
    'AK':'Alaska','AL':'Alabama','AR':'Arkansas','AZ':'Arizona','CA':'California',
    'CO':'Colorado','CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia',
    'HI':'Hawaii','IA':'Iowa','ID':'Idaho','IL':'Illinois','IN':'Indiana',
    'KS':'Kansas','KY':'Kentucky','LA':'Louisiana','MA':'Massachusetts','MD':'Maryland',
    'ME':'Maine','MI':'Michigan','MN':'Minnesota','MO':'Missouri','MS':'Mississippi',
    'MT':'Montana','NC':'North Carolina','ND':'North Dakota','NE':'Nebraska','NH':'New Hampshire',
    'NJ':'New Jersey','NM':'New Mexico','NV':'Nevada','NY':'New York','OH':'Ohio',
    'OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina',
    'SD':'South Dakota','TN':'Tennessee','TX':'Texas','UT':'Utah','VA':'Virginia',
    'VT':'Vermont','WA':'Washington','WI':'Wisconsin','WV':'West Virginia','WY':'Wyoming'
}


st.set_page_config(layout="wide")

st.title("U.S. Wealth and Poverty Analysis")

# File Upload
uploaded_file = st.file_uploader(
    "Upload the dataset (.xlsx or .csv)",
    type=["xlsx", "csv"]
)

if not uploaded_file:
    st.info("Please upload a dataset to begin.")
    st.stop()


# Load Dataset
if uploaded_file.name.lower().endswith(".xlsx"):
    df = pd.read_excel(uploaded_file)
else:
    df = pd.read_csv(uploaded_file)

df.columns = df.columns.str.strip()

#Column Names
df = df.rename(columns={
    "Number in Poverty": "Poverty",
    "Number of Millionaires": "Millionaires",
    "State Popiulation": "Population" 
})

st.write("Dataset Preview:")
st.dataframe(df.head())


#Tabs
tab1, tab2, tab3 = st.tabs([
    "Poverty vs Millionaires",
    "Millionaire Density Map",
    "Poverty Rate"
])


# Tab 1
with tab1:
    st.header("Poverty vs Millionaires by State")

    required = ["State", "Poverty", "Millionaires"]
    if not has_cols(df, required):
        st.error(f"Missing required columns: {required}")
    else:
        states = df["State"].dropna().unique().tolist()
        
        # We need the full state names for a clean multiselect if the 'State' column holds codes
        display_names = [code_to_state.get(code, code) for code in states]

        # Randomize default selection based on states
        default_codes = random.sample(states, 5) if len(states) >= 5 else states
        default_names = [code_to_state.get(code, code) for code in default_codes]

        st.caption("Five states are randomly selected by default each time the app loads.")

        selected_names = st.multiselect(
            "Select states",
            display_names,
            default=default_names
        )
        
        # Map selected names back to codes for filtering
        selected_codes = [code for code, name in code_to_state.items() if name in selected_names]


        if selected_codes:
            sub = df[df["State"].isin(selected_codes)].copy()
            sub["Poverty"] = pd.to_numeric(sub["Poverty"], errors="coerce")
            sub["Millionaires"] = pd.to_numeric(sub["Millionaires"], errors="coerce")
            
            # Display name column for the chart labels
            sub["DisplayName"] = sub["State"].map(code_to_state)

            x = np.arange(len(sub))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - width / 2, sub["Poverty"], width, label="People in Poverty")
            ax.bar(x + width / 2, sub["Millionaires"], width, label="Millionaires")

            ax.set_xticks(x)
            ax.set_xticklabels(sub["DisplayName"], rotation=45, ha='right')
            ax.set_ylabel("Population Count")
            ax.set_title("Poverty vs Millionaires by State")
            ax.legend()
            fig.tight_layout() # Added to prevent cutting off rotated labels

            st.pyplot(fig)

            st.write("""
            This chart compares the number of people living in poverty with the number of
            millionaires across selected states. 
            """)


# TAB 2
with tab2:
    st.header("Millionaire Density by U.S. State")

    required = ["State", "Population", "Millionaires"]
    if not has_cols(df, required):
        st.error(f"Missing required columns: {required}")
    else:
        df2 = df.copy()
        df2["Population"] = pd.to_numeric(df2["Population"], errors="coerce")
        df2["Millionaires"] = pd.to_numeric(df2["Millionaires"], errors="coerce")

        #  Millionaire Density 
        df2["Millionaire_Density"] = safe_div(
            df2["Millionaires"],
            df2["Population"]
        )

        df2["Millionaire_Density"] = df2["Millionaire_Density"].fillna(0)

        # LOG SCALE
        # The 1e-6 is added for states with zero density to avoid error
        df2["Log_Density"] = np.log10(df2["Millionaire_Density"] + 1e-6) 

        # State Codes
        # The 'State' column in the uploaded data already contains the state codes
        # Use it for Plotly's 'locations' argument.
        df2["StateCode"] = df2["State"] 
        
        # Add full state names for hover labels
        df2["StateName"] = df2["StateCode"].map(code_to_state)
        
        # Drop any row where we failed to get a state code or name
        plot_df = df2.dropna(subset=["StateCode", "StateName"]) 

        # Choropleth Map
        fig_map = px.choropleth(
            plot_df,
            locations="StateCode",
            locationmode="USA-states",
            color="Log_Density",
            scope="usa",
            hover_name="StateName", # Use full name for hover display
            hover_data={
                "Population": True,
                "Millionaires": True,
                "Millionaire_Density": ":.6f", # Display density with 6 decimal places
                "StateCode": False # Hide the code in hover
            },
            title="Millionaire Density by U.S. State (Log Scaled)",
            color_continuous_scale="Viridis" # A good color map for log-scaled sequential data
        )

        fig_map.update_layout(
            coloraxis_colorbar=dict(
                title="Log(Millionaire Density)"
            )
        )

        st.plotly_chart(fig_map, use_container_width=True)

        st.write("""
        Since millionare numbers densities are very small I used a log scale
        to make it easier to visualize the differences in each state
        """)
        # 

# TAB 3 
with tab3:
    st.header("Poverty Rate by State")

    required = ["State", "Poverty", "Population"]
    if not has_cols(df, required):
        st.error(f"Missing required columns: {required}")
    else:
        df3 = df.copy()
        df3["Population"] = pd.to_numeric(df3["Population"], errors="coerce")
        df3["Poverty"] = pd.to_numeric(df3["Poverty"], errors="coerce")

        df3["Poverty_Rate"] = safe_div(
            df3["Poverty"],
            df3["Population"]
        )
        
        # Full state names for display
        df3["DisplayName"] = df3["State"].map(code_to_state)


        df_sorted = df3.sort_values(
            "Poverty_Rate",
            ascending=False
        ).dropna(subset=["Poverty_Rate", "DisplayName"])

        fig3, ax3 = plt.subplots(figsize=(10, 10))
        # Plot using the full state names
        ax3.barh(df_sorted["DisplayName"], df_sorted["Poverty_Rate"] * 100)
        ax3.set_xlabel("Poverty Rate (%)")
        ax3.set_ylabel("State")
        ax3.set_title("Poverty Rate by State (Highest to Lowest)")
        ax3.invert_yaxis()
        fig3.tight_layout()

        st.pyplot(fig3)

        st.write("""
        This chart shows the poverty rate by state, with Mississippi at the top and New Hampshire at the bottom. Higher poverty
        rates appear in the southern states with lower rates in the northeast and midwest.
        """)






