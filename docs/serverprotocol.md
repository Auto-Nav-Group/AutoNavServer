# Server Protocol Docs

### Supported POST commands:
- `/update_map` - Updates the map used during server analysis
- `/kill_thread` - Shuts down the server entirely.
### Supported GET commands:
- `/pathfind` - Finds a path between two points
- `/util/status` - Returns the status of the server


## Functions:
Important notices:
parameters will not be required, however if they are not provided, the server will use the previously used values.
### `update_map`:
Updates the map used during server analysis
#### Parameters (HTTP headers):
- None
#### Attachable data (HTTP post data):
- `map` - The map to be used during server analysis
#### Returns:
- None
#### Example python request:
```
from map import Map
import requests

JSON_FILE = 'E:/Some/File/Path/Map.json'

mapobj = Map(JSON_FILE)
requests.post('http://localhost:8000/update_map', data=mapobj.to_json())
```

#

### `pathfind`:
Calculates the fastest path between two points based on the map provided by `update_map`
#### Parameters (HTTP headers):
- `startnode` - The starting node of the path
- `endnode` - The ending node of the path
#### Returns:
- `path` - The path between the two points
- `distance` - The distance of the path
- `time` - The time taken to calculate the path (debug purposes)
#### Example python request:
```
from map import Map
import requests

# Assume that the map has already been updated

requests.get('http://localhost:8000/pathfind', headers={'startnode' : NodeGraph.Node(Geometry.Point(1,1)).to_json(), 'endnode' : NodeGraph.Node(Geometry.Point(5, 5)).to_json()})
```

#

### `kill_thread`:
Shuts down the server entirely. This is a debug command and should not be used in production.
#### Parameters (HTTP headers):
- None
#### Returns:
- None
#### Example python request:
```
requests.post('http://localhost:8000/kill_thread')
```

#

### `util/status`:
Returns the status of the server
#### Parameters (HTTP headers):
- None
#### Returns:
- `status` - The status of the server
#### Example python request:
```
requests.get('http://localhost:8000/util/status')
```

#

### `get_nodegraph`:
Returns the generated nodegraph
#### Parameters (HTTP headers):
- None
#### Returns:
- `edges` - List of edges in the nodegraph
#### Example python request:
```
requests.get('http://localhost:8000/get_nodegraph
```