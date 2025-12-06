# League Metrics Summary

## Start vs Second Win Table
            start_games  start_wins  start_win_rate  second_games  second_wins  second_win_rate
one_head             40          21           0.525            40           19            0.475
n_head               40          24           0.600            40           29            0.725
mtdf                 40          29           0.725            40           39            0.975
alphazero            40          21           0.525            40           17            0.425
dqn_double           40           0           0.000            40            0            0.000

## Forced Move Response
            win_available  win_success  win_rate  block_available  block_success  block_rate
player                                                                                      
alphazero              38           38  1.000000              163            163    1.000000
dqn_double             11            0  0.000000               76              2    0.026316
mtdf                   68           68  1.000000               89             89    1.000000
n_head                 54           53  0.981481               44             36    0.818182
one_head               40           40  1.000000               70             70    1.000000

## Average Turns
            avg_turns  std_turns
player                          
alphazero   23.520779  10.085858
dqn_double  13.206522   3.615924
mtdf        26.871041  11.235727
n_head      28.910696  11.154291
one_head    21.644258   9.175065

## Search Cost
            decision_time_ms  search_nodes
player                                    
alphazero        1406.475858    100.000000
dqn_double        118.865787      0.000000
mtdf             1121.811540   2298.087104
n_head           2623.648220   1987.902388
one_head            4.509110      6.648459

## Plots
- Forced response: plots\forced_response.png
- Average turns: plots\average_turns.png
- Search cost: plots\search_cost.png