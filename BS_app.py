import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import streamlit as st

# =======================================

# Black scholes equation to calculate prices
class BlackScholes():
    def __init__(self, spot_price, strike_price, time_to_maturity, interest_rate, volatility):
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.interest_rate = interest_rate
        self.volatility = volatility

    def calculate_price(self):

        S = self.spot_price
        K = self.strike_price
        T = self.time_to_maturity
        r = self.interest_rate
        sigma = self.volatility

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        d2 = d1 - sigma * np.sqrt(T)

        call_price = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(- r * T)

        put_price = K * np.exp(- r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return call_price, put_price


# =======================================

# Create the grid for the heatmap, plotting the price to its respective spot for the call and put grids and returning it
def create_grid(spot_range, vol_range, strike_price, time_to_maturity, interest_rate):
    call_grid = np.empty((len(vol_range), len(spot_range)))
    put_grid = np.empty((len(vol_range), len(spot_range)))


    for x, vol in enumerate(vol_range):
        for y, spot in enumerate(spot_range):
            bs = BlackScholes(spot, strike_price, time_to_maturity, interest_rate, vol)
            call, put = bs.calculate_price()
            call_grid[x, y] = call
            put_grid[x, y] = put

    return call_grid, put_grid

# =======================================

# Use the previously created call and put price grid to plot a heatmap
def create_heatmap(grid, x_labels, y_labels, title, grid_n):
    
    max_labels = 10
    step = max(1, len(x_labels) // max_labels)
    
    cmap=plt.get_cmap("RdYlGn_r")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    font_size = max(6, 14 - grid_n // 2)
    sns.heatmap(grid, annot=True, cmap=cmap, fmt=".2f", annot_kws={"size": font_size}, ax=ax, square=True, xticklabels=False, yticklabels=y_labels)
    
    ax.set_xticks(np.arange(0.5, len(x_labels), step))
    rotation = -45 if grid_n > 10 else 0
    ax.set_xticklabels(x_labels[::step], rotation=rotation)
    
    ax.invert_yaxis()
    ax.set_xlabel("Spot Price", fontsize=12)
    ax.set_ylabel("Volatility", fontsize=12)
    ax.set_title(f"{title} Prices", fontsize=16)

    return fig
    

# =======================================

def styled_box(text, color):
    st.markdown(f"""
    <div style="
        background-color:{color};
        padding:10px;
        border-radius:10px;
        text-align:center;
        font-size:24px;
        font-weight:bold;
        color:black;">
        {text}
    </div>
    """, unsafe_allow_html=True)


def main(): 
    st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
    )
    
   
    with st.sidebar:
        st.sidebar.header("Option Parameters")
        
        spot_price = st.number_input("Spot Price", value=100.00)
        strike_price = st.number_input("Strike Price", value=100.00)
        time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.00)
        interest_rate = st.number_input("Interest Rate", value=0.05)
        volatility = st.number_input("Volatility", value=0.20)
        
        st.divider()

        st.sidebar.header("Heatmap Parameters")
        
        st.markdown("**Spot Price:**")
        spot_min = st.number_input("Minimum Spot Price", min_value=0.01, value=spot_price * 0.8)
        spot_max = st.number_input("Maximum Spot Price", min_value=0.01, value=spot_price * 1.2)

        st.text("")

        st.markdown("**Volatility:**")
        vol_min, vol_max = st.slider(
            "Volatility Range", 
            min_value=0.01, max_value=1.0, 
            value=(max(0.01, volatility * 0.5), min(1.0, volatility * 1.5))
            )

        if spot_min > spot_max:
            st.error("Minimum Spot Price must be less than Maximum Spot Price.")
            st.stop()
        
        st.text("")

        st.markdown("**Grid Settings:**")
        grid_n = st.sidebar.slider("Grid Density", min_value=2, max_value=25, value=10, step=1)

    
    st.title("Interactive Black-Scholes Visualizer")

    st.subheader("Option Values:")



    spot_range = np.linspace(spot_min, spot_max, num=grid_n)
    vol_range = np.linspace(vol_min, vol_max, num=grid_n)
    call_grid, put_grid = create_grid(spot_range, vol_range, strike_price, time_to_maturity, interest_rate)

    x_labels = [f"{num:.2f}" for num in spot_range]
    y_labels = [f"{num:.2f}" for num in vol_range]

    call_plot = create_heatmap(call_grid, x_labels, y_labels, "Call", grid_n)
    put_plot = create_heatmap(put_grid, x_labels, y_labels, "Put", grid_n)    

    real_call, real_put = BlackScholes(spot_price, strike_price, time_to_maturity, interest_rate, volatility).calculate_price()

  

    col1, col2 = st.columns(2)
    with col1: 
        styled_box(f"Call Value: ${real_call:.2f}", "#2ECC71")
        st.text("")
        st.subheader("Call Price Heatmap")
        st.pyplot(call_plot)

    with col2: 
        styled_box(f"Put Value: ${real_put:.2f}", "#E74C3C")
        st.text("")
        st.subheader("Put Price Heatmap")
        st.pyplot(put_plot)
    

main()