
import heapq

# Define the goal state
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Define the possible moves (Up, Down, Left, Right)
MOVES = [(-3, 'Up'), (3, 'Down'), (-1, 'Left'), (1, 'Right')]

# Heuristic function: Manhattan Distance
def manhattan_distance(state):
    distance = 0
    for i in range(9):
        if state[i] != 0:
            target_row, target_col = divmod(state[i] - 1, 3)
            current_row, current_col = divmod(i, 3)
            distance += abs(target_row - current_row) + abs(target_col - current_col)
    return distance

# A* Algorithm to solve the 8-puzzle problem
def a_star(start_state):
    # Priority Queue (min-heap) for A* algorithm
    open_list = []
    # Set of visited states
    visited = set()
    # Push the initial state into the priority queue with (f, g, state, path)
    heapq.heappush(open_list, (manhattan_distance(start_state), 0, start_state, []))

    while open_list:
        _, cost, current_state, path = heapq.heappop(open_list)

        # If the current state is the goal state, return the solution
        if current_state == GOAL_STATE:
            return path

        # Mark the current state as visited
        visited.add(current_state)

        # Find the blank space (0)
        blank_index = current_state.index(0)
        row, col = divmod(blank_index, 3)

        # Generate the possible moves
        for move, direction in MOVES:
            new_blank_index = blank_index + move

            # Check if the move is valid (don't cross the grid's boundaries)
            if direction == 'Left' and col == 0: continue
            if direction == 'Right' and col == 2: continue
            if direction == 'Up' and row == 0: continue
            if direction == 'Down' and row == 2: continue

            # Swap the blank space (0) with the adjacent tile
            new_state = list(current_state)
            new_state[blank_index], new_state[new_blank_index] = new_state[new_blank_index], new_state[blank_index]
            new_state_tuple = tuple(new_state)

            # Avoid revisiting the same state
            if new_state_tuple not in visited:
                new_cost = cost + 1
                heapq.heappush(open_list, (new_cost + manhattan_distance(new_state_tuple), new_cost, new_state_tuple, path + [direction]))

    return None  # If no solution is found

# Function to print the state of the puzzle
def print_puzzle(state):
    for i in range(0, 9, 3):
        print(state[i:i + 3])
    print()

# Main function to solve the 8-puzzle
def main():
    # Example start state (change this to any other state to test)
    start_state = (5, 6, 2, 8, 0, 3, 4, 7, 1)

    print("Initial Puzzle State:")
    print_puzzle(start_state)

    solution = a_star(start_state)
    if solution:
        print("Solution steps:")
        for step in solution:
            print(step)
    else:
        print("No solution found!")

if __name__ == "__main__":
    main()
