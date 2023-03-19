# Black-Scholes-Merton Model package
# ---------------------------------------------------------------------------------------------------------------------
# Author:       Kevin Kraemer
# Name:         black_scholes_merton
# Created:      01-03-2020
# Last Updated: 28-03-2020
#
# Purpose: European Option pricing
# Source: John C. Hull, "Options, Futures, and Other Derivatives", Ed. 6
# -----------------------------------------------------------------------------

# Cheatsheet: Literal String Interpolation ------------------------------------
# format_spec ::=  [[fill]align][sign][#][0][width][,][.precision][type]
# fill        ::=  <any character>
# align       ::=  "<" | ">" | "=" | "^"
# sign        ::=  "+" | "-" | " "
# width       ::=  integer
# precision   ::=  integer
# type        ::=  "b" | "c" | "d" | "e" | "E" | "f" | "F" | "g" | "G" | "n" | "o" | "s" | "x" | "X" | "%"

# Notes: ----------------------------------------------------------------------
# The value N(-d1) is an area under the normal curve, corresponding to the cumulative
# probability of a value less than or equal to -d1 standard deviations.
# This is the same value as the area above d1, i.e., N(-d1) == 1 - N(d1).
# N(d1) == N(d1)
# -N(d1) == -N(d1)
# N(-d1) == 1 - N(d1)
# -N(-d1) == N(-d1) - 1

# Modules -------------------------------------------------------------------------------------------------------------
from copy import deepcopy
from math import log, sqrt, exp, fabs

import numpy as np
from scipy.stats import norm


# Classes--------------------------------------------------------------------------------------------------------------
# Base class (parent)


class BlackScholesMertonOption:
    """
    Valuation of European Options in Black-Scholes-Merton Model (incl. dividend)
    Attributes:
    -----------
    stock_price: float
        Initial stock/index level (expressed in CCY)
    strike: float
        Strike price (expressed in CCY)
    time_to_maturity: float
        Time to maturity (expressed in years)
    risk_free_rate: float
        Constant risk-free short rate (annualized, assumes flat term structure)
    sigma: float
        Volatility factor (expressed as annualized standard deviation of returns)
    dividend_yield: float
        Dividend Yield (annualized dividend rate, assumes continuous yield)
    Methods:
    --------
    value: float
        return present value of European bsm_option
    delta: float
        return delta of a European bsm_option
    gamma: float
        return gamma of a European bsm_option
    theta: float
        return theta of a European bsm_option
    rho: float
        return rho of a European bsm_option
    vega: float
        return vega of a European bsm_option
    """

    def __init__(self, stock_price: float, strike: float, time_to_maturity: float, risk_free_rate: float, sigma: float,
                 dividend_yield: float = 0.0):
        assert stock_price >= 0.0, "Stock price cannot be less than zero."
        assert strike >= 0.0, "Strike cannot be less than zero."
        assert time_to_maturity >= 0.0, "Time to maturity cannot be less than zero."
        assert sigma >= 0.0, "Volatility cannot be less than zero."
        assert dividend_yield >= 0.0, "Dividend yield cannot be less than zero."
        self.s = float(stock_price)
        self.k = float(strike)
        self.t = float(time_to_maturity)
        self.rf = float(risk_free_rate)
        self.sigma = float(sigma)
        self.div = float(dividend_yield)

    def __repr__(self) -> str:
        return f"{', '.join([f'{k}: {v}' for k, v in self.__dict__.items()])}"

    def __str__(self) -> str:
        return f"BlackScholesMertonOption({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])})"

    def d1(self) -> float:
        return (log(self.s / self.k) + (self.rf - self.div + 0.5 * self.sigma ** 2) * self.t) / \
            (self.sigma * sqrt(self.t))

    def d2(self) -> float:
        return (log(self.s / self.k) + (self.rf - self.div - 0.5 * self.sigma ** 2) * self.t) / \
            (self.sigma * sqrt(self.t))

    def pdf_d1(self) -> float:
        """Returns n(d1)."""
        return self._pdf(self.d1())

    def pdf_d2(self) -> float:
        """Returns n(d2)."""
        return self._pdf(self.d2())

    def cdf_d1(self) -> float:
        """Returns N(d1)."""
        return self._cdf(self.d1())

    def cdf_d2(self) -> float:
        """Returns N(d2)."""
        return self._cdf(self.d2())

    def value(self) -> float:
        print("Cannot calculate bsm_option value for base class BlackScholesMertonOption.")
        return 0.0

    def delta(self) -> float:
        print("Cannot calculate the bsm_option greek delta for base class BlackScholesMertonOption.")
        return 0.0

    def gamma(self) -> float:
        print("Cannot calculate the bsm_option greek gamma for base class BlackScholesMertonOption.")
        return 0.0

    def theta(self) -> float:
        print("Cannot calculate the bsm_option greek theta for base class BlackScholesMertonOption.")
        return 0.0

    def rho(self) -> float:
        print("Cannot calculate the bsm_option greek rho for base class BlackScholesMertonOption.")
        return 0.0

    def vega(self) -> float:
        print("Cannot calculate the bsm_option greek vega for base class BlackScholesMertonOption.")
        return 0.0

    @staticmethod
    def _pdf(x: float, mean: float = 0.0, std_dev: float = 1.0) -> float:
        """Return the value of the probability distribution function. n(x)"""
        return norm.pdf(x, mean, std_dev)

    @staticmethod
    def _cdf(x: float, mean: float = 0.0, std_dev: float = 1.0) -> float:
        """Return the value of the cumulative distribution function. steps(x)"""
        return norm.cdf(x, mean, std_dev)


# Derived class (child)


class BSMCallOption(BlackScholesMertonOption):
    def __init__(self, stock_price: float, strike: float, time_to_maturity: float, risk_free_rate: float, sigma: float,
                 dividend_yield: float = 0.0):
        super().__init__(stock_price, strike, time_to_maturity, risk_free_rate, sigma, dividend_yield)

    def __repr__(self):
        return f"BSMCallOption\n-------------\nParams: {', '.join([f'{k}={v}' for k, v in self.__dict__.items()])}\n" \
               f"Value: {self.value()}"

    def __str__(self):
        return f"BSMCallOption({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])})"

    def value(self) -> float:
        """Return the value of a European Call Option."""
        return self.s * exp(-self.div * self.t) * self.cdf_d1() - self.k * exp(-self.rf * self.t) * self.cdf_d2()

    def delta(self) -> float:
        """Return the delta of a European Call Option."""
        return exp(-self.div * self.t) * self.cdf_d1()

    def gamma(self) -> float:
        """Return the gamma of a European Call  Option."""
        return (exp(-self.div * self.t) * self.pdf_d1()) / (self.s * self.sigma * sqrt(self.t))

    def theta(self) -> float:
        """Return the theta of a European Call  Option."""
        theta = -self.s * self.pdf_d1() * self.sigma * exp(-self.div * self.t) / (
                (2 * sqrt(self.t)) + self.div * self.s * self.cdf_d1() * exp(-self.div * self.t) -
                self.rf * self.k * exp(-self.rf * self.t) * self.cdf_d2()
        )
        return theta

    def rho(self) -> float:
        """Return the rho of a European Call Option."""
        return self.k * self.t * exp(-self.rf * self.t) * self.cdf_d2()

    def vega(self) -> float:
        """Return the vega of a European Option."""
        return self.s * sqrt(self.t) * self.pdf_d1() * exp(-self.div * self.t)

    def summary(self) -> str:
        """Return the summary of a European Call Option."""
        summary = (
            f"Summary\n"
            f"-------------\n"
            f"value: {self.value()}\n"
            f"delta: {self.delta()}\n"
            f"gamma: {self.gamma()}\n"
            f"theta: {self.theta()}\n"
            f"rho:   {self.rho()}\n"
            f"vega:  {self.vega()}"
        )
        return summary


# Derived class (child)


class BSMPutOption(BlackScholesMertonOption):
    def __init__(self, stock_price: float, strike: float, time_to_maturity: float, risk_free_rate: float, sigma: float,
                 dividend_yield: float = 0.0):
        super().__init__(stock_price, strike, time_to_maturity, risk_free_rate, sigma, dividend_yield)

    def __repr__(self) -> str:
        return f"BSMPutOption\n------------\nParams: {', '.join([f'{k}={v}' for k, v in self.__dict__.items()])}\n" \
               f"Value: {self.value()}"

    def __str__(self) -> str:
        return f"BSMPutOption({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])})"

    def value(self) -> float:
        """Return the value of a European Put Option."""
        return self.k * exp(-self.rf * self.t) * (1 - self.cdf_d2()) - self.s * exp(self.div * self.t) * \
            (1 - self.cdf_d1())

    def delta(self) -> float:
        """Return the delta of a European Put Option."""
        return exp(-self.div * self.t) * (self.cdf_d1() - 1)

    def gamma(self) -> float:
        """Return the gamma of a European Put Option."""
        return (exp(-self.div * self.t) * self.pdf_d1()) / (self.s * self.sigma * sqrt(self.t))

    def theta(self) -> float:
        """Return the theta of a European Put Option."""
        theta = -self.s * self.pdf_d1() * self.sigma * exp(-self.div * self.t) / (
                (2 * sqrt(self.t)) - self.div * self.s * (1 - self.cdf_d1()) * exp(-self.div * self.t) +
                self.rf * self.k * exp(-self.rf * self.t) * (1 - self.cdf_d2())
        )
        return theta

    def rho(self) -> float:
        """Return the rho of a European Put Option."""
        return -self.k * self.t * exp(-self.rf * self.t) * (1 - self.cdf_d2())

    def vega(self) -> float:
        """Return the vega of a European Option."""
        return self.s * sqrt(self.t) * self.pdf_d1() * exp(-self.div * self.t)

    def summary(self) -> str:
        """Return the summary of a European Put Option."""
        summary = (
            f"Summary\n"
            f"-------------\n"
            f"value: {self.value()}\n"
            f"delta: {self.delta()}\n"
            f"gamma: {self.gamma()}\n"
            f"theta: {self.theta()}\n"
            f"rho:   {self.rho()}\n"
            f"vega:  {self.vega()}"
        )
        return summary


# Functions


def implied_volatility(bsm_option: BlackScholesMertonOption, option_market_price: float, method: str = "Bisection"):
    """
    Return the implied volatility from observed stock prices and bsm_option prices in the market.
    Source
    ------
    https://medium.com/hypervolatility/extracting-implied-volatility-newton-raphson-secant-and-bisection-approaches-fae83c779e56
    """

    def implied_volatility_simple(option: BlackScholesMertonOption, market_price: float) -> float:
        """Return the implied volatility using a simple iterative method."""
        option.sigma = 0.001
        tolerance = 1e-5
        while option.sigma < 1000:
            if market_price - option.value() < tolerance:
                break
            option.sigma += 0.001
        return option.sigma

    def implied_volatility_newton_raphson(option: BlackScholesMertonOption, market_price: float) -> float:
        """Return the implied volatility using the Newton Raphson Method."""
        option.sigma = 1.0
        iteration, iteration_max = 0, 1000
        tolerance = 1e-10
        while iteration < iteration_max and fabs(option.value() - market_price) > tolerance:
            option.sigma = option.sigma - (option.value() - market_price) / option.vega()
        iteration += 1
        return option.sigma

    def implied_volatility__bisection(option: BlackScholesMertonOption, market_price: float) -> float:
        """Return the implied volatility using the Bisection method."""
        sigma_low, sigma_high = 0.001, 1000
        iteration, iteration_max = 0, 1000
        tolerance = 1e-10
        option_low = deepcopy(option)
        while iteration < iteration_max and fabs(market_price - option.value()) > tolerance:
            option_low.sigma, option.sigma = sigma_low, sigma_high
            option.sigma = sigma_low + (market_price - option_low.value()) * (sigma_high - sigma_low) / (
                    option.value() - option_low.value()
            )
            if option.value() < market_price:
                sigma_low = option.sigma
            elif option.value() >= market_price:
                sigma_high = option.sigma
            iteration += 1
        return option.sigma

    def implied_volatility_secant(option: BlackScholesMertonOption, market_price: float) -> float:
        """Return the implied volatility using the Secant method."""
        from copy import deepcopy
        sigma_low, sigma_high = 0.001, 1000
        iteration, iteration_max = 0, 1000
        tolerance = 1e-10
        option_low = deepcopy(option)
        while iteration < iteration_max and fabs(market_price - option.value()) > tolerance:
            option_low.sigma, option.sigma = sigma_low, sigma_high
            option.sigma = option.sigma + (market_price - option_low.value() * (option_low.sigma - option.sigma)) / (
                    option_low.value() - option.value()
            )
            iteration += 1
        return option.sigma

    switch_case = {
        "Bisection": implied_volatility__bisection,  # Good for American-style Options, Medium-Fast
        "NewtonRaphson": implied_volatility_newton_raphson,  # Good for European-style Options, Fast, Sensitive to vega
        "Secant": implied_volatility_secant,  # Good for European-style Options, no need to calculate vega
        "Simple": implied_volatility_simple,  # Simple, slow and not precise compared to above methods
    }
    return switch_case.get(method, lambda: "Invalid Method.")(bsm_option, option_market_price)


def print_option_values_and_deltas(call_option: BSMCallOption, put_option: BSMPutOption, stock_price: float,
                                   steps: int = 10):
    header = f"{'Price':<8} | {'Call':<8} | {'C Delta':<8} | {'Put':<8} | {'P Delta':<8}"
    print(header)
    print("-" * len(header))
    for stock_price in np.arange(stock_price - steps, stock_price + steps + 1, 1):
        call_option.s, put_option.s = stock_price, stock_price
        print(
            f"{stock_price:>8.2f} | "
            f"{call_option.value():>8.2f} | "
            f"{call_option.delta():>8.4f} | "
            f"{put_option.value():>8.2f} | "
            f"{put_option.delta():>8.4f}"
        )
