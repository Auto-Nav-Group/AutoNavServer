# Server Protocol Docs

### Supported POST commands:
- `update_map` - Updates the map used during server analysis
### Supported GET commands:
- `pathfind` - Finds a path between two points


## Functions:
Important notices:
parameters will not be required, however if they are not provided, the server will use the previously used values.
### `update_map`:
Updates the map used during server analysis
#### Parameters (HTTP headers):
- Command - `update_map`:
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
requests.post('http://localhost:8000', headers={'Command' : 'update_map'}, data=mapobj.to_json())
```

#

### `pathfind`:
Calculates the fastest path between two points based on the map provided by `update_map`
#### Parameters (HTTP headers):
- Command - The command to be run: `pathfind`:
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

requests.get('http://localhost:8000', headers={'Command' : 'pathfind', 'startnode' : NodeGraph.Node(Geometry.Point(1,1)).to_json(), 'endnode' : NodeGraph.Node(Geometry.Point(5, 5)).to_json()})
```

#

### `kill_server`:
Kills the HTTP server responsible for running the server protocol. WILL CLOSE CONNECTION!!!