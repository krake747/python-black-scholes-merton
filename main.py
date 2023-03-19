# Main Program --------------------------------------------------------------------------------------------------------
from black_scholes_merton import BlackScholesMertonOption, BSMCallOption, BSMPutOption, print_option_values_and_deltas,\
    implied_volatility


def main() -> None:
    print("Executing as main program: BSM Option Valuation.")

    bsm = BlackScholesMertonOption(
        stock_price=100,
        strike=100,
        time_to_maturity=0.12876712328767123,
        risk_free_rate=0.2,
        sigma=0.2
    )

    print(bsm.value(), "\n")

    call_option = BSMCallOption(
        stock_price=100,
        strike=100,
        time_to_maturity=0.12876712328767123,
        risk_free_rate=0.2,
        sigma=0.2
    )

    print(repr(call_option), "\n")
    print(call_option.summary(), "\n")

    put_option = BSMPutOption(
        stock_price=100,
        strike=100,
        time_to_maturity=0.12876712328767123,
        risk_free_rate=0.2,
        sigma=0.2
    )

    print(repr(put_option))
    print(put_option.summary(), "\n")

    print_option_values_and_deltas(call_option, put_option, 100, 10)

    implied_volatility_result = implied_volatility(put_option, 8, method="NewtonRaphson")
    print(implied_volatility_result)


# Start of Program ----------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
