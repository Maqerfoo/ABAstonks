import pandas as pd
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
from IPython import display
import time


plt.ion()
style.use('ggplot')
register_matplotlib_converters()

VOLUME_CHART_HEIGHT = 0.33

class BitcoinTradingGraph:
    """A Bitcoin trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, df):
        df=df.reset_index(level=0)# ok
        df["Date"]=pd.to_datetime(df['Date']).apply(lambda x: x.date())
        self.df = df

        self.df['Time'] = self.df['Date'] #.apply(

        self.df = self.df.sort_values('Time')



    def _render_net_worth(self, step_range, dates, current_step, net_worths, benchmarks,trades_btc,trades_gold,net_worth,initial_net_worth,profit_percent):
       
        plt.clf()

        fig = plt.figure()
        fig.suptitle('Net worth: $' + str(net_worth) + ' | Profit: ' + str(profit_percent) + '%')
           
        

        # Plot 1st subplot - net worths
        

        net_worth_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)
        net_worth_ax.cla()
        net_worth_ax.plot(dates, net_worths[step_range], label='Net Worth', color="g")

        # benchmarks 

        colors = ['orange', 'cyan', 'purple', 'blue', 'magenta', 'yellow', 'black', 'red', 'green']

        for i, benchmark in enumerate(benchmarks):
            net_worth_ax.plot(dates, benchmark['values'][step_range], label=benchmark['label'], color=colors[i % len(colors)], alpha=0.3)
  

        # Show legend, which uses the label we defined for the plot above
        net_worth_ax.legend()
        legend = net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = self.df['Time'].values[current_step]
        last_net_worth = net_worths[current_step]
        
        # Annotate the current net worth on the net worth graph
        net_worth_ax.annotate('{0:.2f}'.format(last_net_worth), (last_date, last_net_worth),
                                   xytext=(last_date, last_net_worth),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        # Add space above and below min/max net worth
        net_worth_ax.set_ylim(min(net_worths) / 1.25, max(net_worths) * 1.25)




        
        # Plot 2nd subplot - net worths
        # Create bottom subplot for shared price/volume axis
        
        price_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=8, colspan=1) #sharex=net_worth_ax)
        price_ax .cla()
        price_ax.plot(dates, self.df['Close_BTC'].values[step_range], color="black")

        last_date = self.df['Time'].values[current_step]
        last_close = self.df['Close_BTC'].values[current_step]
        last_high = self.df['High_BTC'].values[current_step]

        # Print the current price to the price axis
        price_ax.annotate('{0:.2f}'.format(last_close), (last_date, last_close),
                               xytext=(last_date, last_high),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")
        ylim = price_ax.get_ylim()
        price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0])* VOLUME_CHART_HEIGHT, ylim[1])

        
        
        price_ax2 = price_ax.twinx()
        price_ax2.cla()
                
        price_ax2.plot(dates, self.df['Close_GOLD'].values[step_range],   color='magenta')

        last_date2 = self.df['Time'].values[current_step]
        last_close2 = self.df['Close_GOLD'].values[current_step]
        last_high2 = self.df['High_GOLD'].values[current_step]

        # Print the current price to the price axis
        price_ax2.annotate('{0:.2f}'.format(last_close2), (last_date2, last_close2),
                               xytext=(last_date2, last_high2),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")
        
        for trade in trades_btc:
                    if trade['step'] in range(sys.maxsize)[step_range]:
                        date = self.df['Time'].values[trade['step']]
                        close = self.df['Close_BTC'].values[trade['step']]

                        if trade['type'] == 'buy':
                            color = 'g'
                        else:
                            color = 'r'

                        price_ax.annotate(' ', (date, close),
                                              xytext=(date, close),
                                              size="large",
                                              arrowprops=dict(arrowstyle='simple', facecolor=color))



        




        # Plot 2nd subplot - net worths
        volume1 = np.array(self.df['Volume_BTC'].values[step_range])
        volume_ax1 = price_ax.twinx()
        volume_ax1.cla()
                
        volume_ax1.plot(dates, volume1,  color='blue')
        volume_ax1.fill_between(dates, volume1, color='blue', alpha=0.5)
        volume_ax1.set_ylim(0, max(volume1) / VOLUME_CHART_HEIGHT)
        volume_ax1.yaxis.set_ticks([])

        volume2 = np.array(self.df['Volume_GOLD'].values[step_range])
        volume_ax2 = price_ax.twinx()
        volume_ax2.cla()
                
        volume_ax2.plot(dates, volume2,  color='yellow')
        volume_ax2.fill_between(dates, volume2, color='yellow', alpha=0.5)
        volume_ax2.set_ylim(0, max(volume2) / VOLUME_CHART_HEIGHT)
        volume_ax2.yaxis.set_ticks([])

        for trade in trades_gold:
                    if trade['step'] in range(sys.maxsize)[step_range]:
                        date = self.df['Time'].values[trade['step']]
                        close = self.df['Close_GOLD'].values[trade['step']]

                        if trade['type'] == 'buy':
                            color = 'g'
                        else:
                            color = 'r'

                        price_ax2.annotate(' ', (date, close),
                                              xytext=(date, close),
                                              size="large",
                                              arrowprops=dict(arrowstyle='simple', facecolor=color))
        
        date_labels = self.df['Date'].values[step_range]

        price_ax.set_xticklabels(date_labels, rotation=45, horizontalalignment='right')
       # Hide duplicate net worth date labels
        plt.setp(net_worth_ax.get_xticklabels(), visible=False)

        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)

        # plt.show(block=False)
        display.clear_output(wait=True)
        display.display(plt.gcf())

        display.clear_output(wait=True)
        
        # Add padding to make graph easier to view
        # 

        time.sleep(2)
        plt.pause(0.1)

    def render(self, current_step, net_worths, benchmarks, trades_btc,trades_gold, window_size=200):
        
        
        net_worth = round(net_worths[-1], 2)
        initial_net_worth = round(net_worths[0], 2)
        profit_percent = round((net_worth - initial_net_worth) / initial_net_worth * 100, 2)
        

        window_start = max(current_step - window_size, 0)
        step_range = slice(window_start, current_step + 1)
        dates = self.df['Time'].values[step_range]

        self._render_net_worth(step_range, dates, current_step, net_worths, benchmarks,trades_btc,trades_gold,net_worth,initial_net_worth,profit_percent)
       
        # Necessary to view frames before they are unrendered
        #
    def close(self):
        plt.close()





