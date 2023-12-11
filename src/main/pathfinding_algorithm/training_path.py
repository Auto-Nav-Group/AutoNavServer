import heapq

def topKey(queue):
    queue.sort()
    if len(queue) > 0:
        return queue[0][:2]
    else:
        return None


def heuristic(graph, state, id):
    x_dist = abs(int(id.split('x')[1][0]) - int(state.split('x')[1][0]))
    y_dist = abs(int(id.split('y')[1][0]) - int(state.split('y')[1][0]))
    return max(x_dist, y_dist)


def calc_key(graph, id, current_state, k_m):
    return min(graph.graph[id].g, graph.graph[id].rhs) + heuristic(graph, id, current_state) + k_m, min(graph.graph[id].g, graph.graph[id].rhs)


def update_vertex(graph, queue, id, current_state, k_m):
    s_goal = graph.goal
    if id != s_goal:
        min_rhs = float('inf')
        for i in graph.graph[id].children:
            min_rhs = min(min_rhs, graph.graph[i].g + graph.graph[id].children[i])
        graph.graph[id].rhs = min_rhs
    id_in_queue = [item for item in queue if id in item]
    if id_in_queue:
        if len(id_in_queue) != 1:
            raise ValueError("")





