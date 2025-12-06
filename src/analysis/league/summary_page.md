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
- Forced response: src\analysis\plots\forced_response.png
- Average turns: src\analysis\plots\average_turns.png
- Search cost: src\analysis\plots\search_cost.png

## Interpretation
- 단일 룩어헤드(one_head)는 매 턴 패턴 창 전체를 평가해 즉시 승/차단을 100% 맞추지만, 평균 승률은 50%대에 머무른다.
- N-step 룩어헤드는 깊이 3까지 순수 미니맥스를 돌리느라 강제 수를 따로 감지하지 않고 휴리스틱 점수만 보고 선택하기 때문에, 강제 차단률이 약 82%로 떨어진다.
- 그럼에도 더 긴 탐색 덕분에 선공 60%, 후공 72.5%로 one_head보다 전반적인 승률이 높으므로, 즉각 전술만 보면 one_head가 낫지만 전체 게임 성능은 n_head가 우위다.
