def create_submission_agent():
    """
    Create lightweight agent for Kaggle submission
    Uses simplified MCTS with heuristic evaluation
    """
    
    def evaluate_position(board, player, rows=6, cols=7):
        """Simple heuristic evaluation"""
        board_array = np.array(board).reshape(rows, cols)
        score = 0
        
        # Center control
        center_col = cols // 2
        for r in range(rows):
            if board_array[r, center_col] == player:
                score += 3
        
        # Check threats and opportunities
        # (simplified for submission size)
        
        return score
    
    def get_valid_actions(board, cols=7):
        board_array = np.array(board).reshape(-1, cols)
        return [c for c in range(cols) if board_array[0, c] == 0]
    
    def apply_action(board, action, player, rows=6, cols=7):
        board_array = np.array(board).reshape(rows, cols).copy()
        for r in range(rows - 1, -1, -1):
            if board_array[r, action] == 0:
                board_array[r, action] = player
                break
        return board_array.flatten().tolist()
    
    def simple_mcts_search(board, player, num_sims=30):
        """Simplified MCTS for submission"""
        valid_actions = get_valid_actions(board)
        
        if not valid_actions:
            return 0
        
        action_scores = {a: 0 for a in valid_actions}
        
        for _ in range(num_sims):
            for action in valid_actions:
                new_board = apply_action(board, action, player)
                score = evaluate_position(new_board, player)
                action_scores[action] += score
        
        # Select best action
        best_action = max(action_scores.items(), key=lambda x: x[1])[0]
        return best_action
    
    def my_agent(observation, configuration):
        import random
        import numpy as np
        
        board = observation.board
        player = observation.mark
        
        # Simple MCTS search
        action = simple_mcts_search(board, player, num_sims=20)
        
        return int(action)
    
    return my_agent

my_agent = create_submission_agent()
