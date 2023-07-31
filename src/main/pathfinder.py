from src.main.nodegraph import NodeGraph
import time
class PathFinder:
    def __init__(self, nodegraph):
        self.path = None
        self.shortestdist = None
        self.nodegraph = nodegraph

    def packagerecalcresults(self, res):
        path = []
        for node in res[0]:
            path.append(node.toJSON())
        return {
            'path' : path,
            'distance' : res[1],
            'time' : res[2]
        }

    def recalculate(self, nodegraph, startnode, endnode):
        """
        Recalculates the path from startnode to endnode using the Dijkstra algorithm
        :param nodegraph: The nodegraph to use
        :param startnode: The start node
        :param endnode: The end node
        :return: The path from startnode to endnode. The distance of that path. Time spent calculating
        """
        """
        Dijkstras algorithm description:
        1) Create distances that start as infinity. Represent the distance from the start location to each node
        2) Loop until all nodes are visited. Unvisited nodes are found by the min distance from the start node
        3) The node discovered by the min distance is visited and explored. Algorithm always checks if it is along the path with the lowest distance
        4) Neighboring nodes are explored. We calculate their distances and check against the minimum distance
        5) Repeat until all nodes are visited
        6) Path is recreated by tracing back from the end node to the start node using the nodes by checking which node's neighbor has a distance equal to the current nodes distance minus that of the connected edge
        """
        if nodegraph is not None:
            self.nodegraph = nodegraph
        if (startnode is None or endnode is None) or (startnode == endnode):
            return None
        if startnode not in nodegraph.nodes:
            nodegraph.addNode(startnode)
        if endnode not in nodegraph.nodes:
            nodegraph.addNode(endnode)
        nodegraph.createJSON('G:\\Projects\\AutoNav\\AutoNavServer\\assets\\testing\\PathfindingNodeGraphDebug.json')
        starttime = time.time()
        # Distances are infinity for all nodes except the start node
        distances = {node.Loc: float('inf') for node in nodegraph.nodes}
        distances[startnode.Loc] = 0

        # Predecessors to store previous nodes
        predecessors = {node.Loc: None for node in self.nodegraph.nodes}

        # List to keep track of visited and unvisited nodes
        visited = set()

        while len(visited) < len(nodegraph.nodes):
            # Find the node with the minimum distance from the start_node
            min_distance = float('inf')
            current_node = None
            for node in self.nodegraph.nodes:
                if node.Loc not in visited and distances[node.Loc] < min_distance:
                    min_distance = distances[node.Loc]
                    current_node = node

            # If there are no reachable nodes left, break the loop
            if current_node is None:
                break

            visited.add(current_node.Loc)

            # Explore neighboring nodes
            for neighbor in current_node.Edges:
                neighbor_node = neighbor.otherloc(current_node.Loc)
                new_distance = distances[current_node.Loc] + neighbor.weight

                if new_distance < distances[neighbor_node]:
                    distances[neighbor_node] = new_distance
                    predecessors[neighbor_node] = current_node

        if predecessors[endnode.Loc] is None:
            return [], float('inf'), 0  # Return an empty path and infinite distance
        # Recreate path from start_node to end_node
        path = []
        current = endnode
        while current != startnode:
            path.append(current)
            current = predecessors[current.Loc]
        path.append(startnode)
        path.reverse()

        self.shortestdist = distances[endnode.Loc]
        self.path = path

        print("Found fastest path in " + str(round(time.time()-starttime, 3)) + " seconds. Segments:" + str(len(path)-1) + " Distance: " + str(distances[endnode.Loc]))
        return path, distances[endnode.Loc], round(time.time()-starttime, 3)
    def DEBUG_benchmarkrecalculate(self):
        pathlengths = []
        times = []
        distances = []
        successful = 0
        failed = 0
        for i in range(len(self.nodegraph.nodes)):
            for j in range(i + 1, len(self.nodegraph.nodes)):
                if i != j:
                    try:
                        d_Path, d_Distance, d_Time = self.recalculate(self.nodegraph, self.nodegraph.nodes[i],
                                                                      self.nodegraph.nodes[j])
                        if len(d_Path)-1 == 0 or d_Distance == float('inf'):
                            failed += 1
                        else:
                            pathlengths.append(len(d_Path) - 1)
                            times.append(d_Time)
                            distances.append(d_Distance)
                            successful += 1
                    except Exception as e:
                        failed += 1
                        print(
                            f"Failed to find path from {self.nodegraph.nodes[i].Loc.unpack()} to {self.nodegraph.nodes[j].Loc.unpack()}")
                        print("Exception:", e)
        print(
            "***DEBUG***" + "\n",
            "Average path length: " + str(sum(pathlengths)/len(pathlengths)) + "\n",
            "Average time: " + str(round(sum(times)/len(times), 4)) + "\n",
            "Average distance: " + str(sum(distances)/len(distances)) + "\n",
            "Successful: " + str(successful) + "\n",
            "Failed: " + str(failed) + "\n",
            "Total nodes: " + str(len(self.nodegraph.nodes)) + "\n",
            "***DEBUG***"
        )