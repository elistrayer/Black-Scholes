import scipy.stats as spi
import numpy as np

class BlackScholes():
    def __init__(self, spot_price, strike_price, time_to_maturity, interest_rate, volatility):
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.interest_rate = interest_rate
        self.volatility = volatility
    
    def calculate_price(self):

        spot_price = self.spot_price
        strike_price = self.strike_price
        time_to_maturity = self.time_to_maturity
        interest_rate = self.interest_rate
        volatility = self.volatility

        d1 = (
            np.log(spot_price / strike_price)
              + (interest_rate + volatility ** 2 / 2) * time_to_maturity
        ) / (
                volatility * np.sqrt(time_to_maturity)
        )

        d2 = d1 - volatility * np.sqrt(time_to_maturity)

        call_price = (
            spi.norm.cdf(d1) * spot_price
            - spi.norm.cdf(d2) * strike_price * np.exp(- interest_rate * time_to_maturity)
        )

        put_price = (
            strike_price * np.exp(- interest_rate * time_to_maturity) * spi.norm.cdf(-d2)
            - spot_price * spi.norm.cdf(-d1)
        )

        return call_price, put_price
    

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

