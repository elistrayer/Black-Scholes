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

        d1, d2 = self.get_d1d2()

        call_price = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(- r * T)

        put_price = K * np.exp(- r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return call_price, put_price
    

    def get_d1d2(self):
        S = self.spot_price
        K = self.strike_price
        T = self.time_to_maturity
        r = self.interest_rate
        sigma = self.volatility

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return d1, d2

    def greeks(self):
        d1, d2 = self.get_d1d2()

        S = self.spot_price
        K = self.strike_price
        T = self.time_to_maturity
        r = self.interest_rate
        sigma = self.volatility

        call_delta = norm.cdf(d1) 
        put_delta = -norm.cdf(-d1)

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
        vega_raw = S * norm.pdf(d1) * np.sqrt(T)
        vega_1pct = vega_raw / 100.0

        call_theta_yr = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        put_theta_yr = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        call_theta_day = call_theta_yr / 365.0
        put_theta_day = put_theta_yr / 365.0

        call_rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        call_rho_1pct = call_rho / 100.0
        put_rho_1pct = put_rho / 100.0

        return {
            "call_delta": call_delta,
            "put_delta": put_delta,
            "gamma": gamma,
            "vega": vega_raw,
            "vega_1pct": vega_1pct,
            "call_theta_yr": call_theta_yr,
            "put_theta_yr": put_theta_yr,
            "call_theta_day": call_theta_day,
            "put_theta_day": put_theta_day,
            "call_rho": call_rho,
            "put_rho": put_rho,
            "call_rho_1pct": call_rho_1pct,
            "put_rho_1pct": put_rho_1pct
        }




def main():
    spot_price = float(input("What is the spot price?: "))
    strike_price= float(input("What is the strike price?: "))
    time_to_maturity = float(input("What is the time to maturity?: "))
    interest_rate = float(input("What is the interest rate?: "))
    volatility = float(input("What is the volatility?: "))

    black_scholes = BlackScholes(spot_price, strike_price, time_to_maturity, interest_rate, volatility)
    call_price, put_price = black_scholes.calculate_price()
    print(f"The call price is ${call_price:.2f}, and the put price is ${put_price:.2f}")


if __name__ == "__main__":
    main()

