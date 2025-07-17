import numpy as np
import plotly.express as px
from scipy.stats import norm

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


    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs = BlackScholes(spot, strike_price, time_to_maturity, interest_rate, vol)
            call, put = bs.calculate_price()
            call_grid[i, j] = call
            put_grid[i, j] = put

    return call_grid, put_grid

# =======================================

# Use the previously created call and put price grid to plot a heatmap
def create_heatmap(grid, x_labels, y_labels, type):
    fig = px.imshow(
        grid,
        text_auto=".2f",
        labels=dict(x="Spot Price", y="Volatility"),
        x=x_labels,
        y=y_labels,
        origin="lower",
        color_continuous_scale="RdYlGn",
    )

    fig.update_layout(
        title = dict(
            text = type,
            x = 0.5,
            xanchor = "center",
            font=dict(size=20)
        )
    )
    fig.update_xaxes(side="bottom", tickangle=0)
    fig.update_yaxes(tickangle=-90)
    
    return fig

# =======================================

def main():
    spot_price = 100
    strike_price = 100
    time_to_maturity = 1
    interest_rate = 0.05
    volatility = 0.2


    spot_range = np.linspace(spot_price * 0.8, spot_price * 1.2, num=10)
    vol_range = np.linspace(volatility * 0.5, volatility * 1.5, num=10)
    call_grid, put_grid = create_grid(spot_range, vol_range, strike_price, time_to_maturity, interest_rate)

    x_labels = [f"{num:.2f}" for num in spot_range]
    y_labels = [f"{num:.2f}" for num in vol_range]

    
    fig = create_heatmap(call_grid, x_labels, y_labels, "Call")
    fig2 = create_heatmap(put_grid, x_labels, y_labels, "Put")

    fig.show()
    fig2.show()


if __name__ == "__main__":
    main()