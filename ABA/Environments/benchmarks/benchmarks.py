from enum import Enum


class SIGNALS(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

def trade_strategy(prices_BTC, prices_GOLD, initial_balance, commission, signal_fn):
    net_worths = [initial_balance]
    balance = initial_balance
    amount_held_BTC = 0
    amount_held_GOLD = 0

    for i in range(1, len(prices_BTC)):
        if (amount_held_BTC > 0) and (amount_held_GOLD>0):
            net_worths.append(balance + amount_held_BTC* prices_BTC[i]+amount_held_GOLD* prices_GOLD[i])
        else:
            net_worths.append(balance)

        signal = signal_fn(i)
        
        if signal == SIGNALS.BUY and amount_held_BTC == 0 and amount_held_GOLD==0:
            amount_held_BTC = (0.5*balance) / (prices_BTC[i] * (1 + commission))
            amount_held_GOLD = (0.5*balance) / (prices_GOLD[i] * (1 + commission))
            balance = 0

    return net_worths

def buy_and_hold (prices_BTC, prices_GOLD, initial_balance, commission):
    def signal_fn(i):
        return SIGNALS.BUY

    return trade_strategy (prices_BTC, prices_GOLD, initial_balance, commission, signal_fn)



