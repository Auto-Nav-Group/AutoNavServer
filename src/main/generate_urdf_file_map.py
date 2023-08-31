from map import Map
import os


ASSET_PATH = "G:\\Projects\\AutoNav\\AutoNavServer\\assets\\drl\\generate_map_files"

def from_map(map):
    for file in os.listdir(ASSET_PATH):
        if file.endswith(".urdf"):
            os.remove(os.path.join(ASSET_PATH, file))

    count = 1
    for obstacle in map.obstacles:
        o = open(os.path.join(ASSET_PATH, "obs_" + str(count) + ".urdf"), "w")
        x = obstacle.Loc.x+obstacle.Size.width/2- map.size.width/2
        y = obstacle.Loc.y + obstacle.Size.height/2 - map.size.height/2
        o.write(
            "<?xml version=\"1.0\"?>\n"
            "<robot name=\"obs_" + str(count) + "\">\n"
            "  <link name=\"obs_" + str(count) + "\">\n"
            "    <collision>\n"
            "      <origin rpy=\"0 0 0\" xyz=\"" + str(x) + " " + str(y) + " 0\"/>\n"
            "      <geometry>\n"
            "        <box size=\"" + str(obstacle.Size.width) + " " + str(obstacle.Size.height) + " 5\"/>\n"                                     
            "      </geometry>\n"
            "    </collision>\n"
            "    <visual>\n"
            "      <origin rpy=\"0 0 0\" xyz=\"" + str(x) + " " + str(y) + " 0\"/>\n"
            "      <geometry>\n"
            "        <box size=\"" + str(obstacle.Size.width) + " " + str(obstacle.Size.height) + " 5\"/>\n"
            "      </geometry>\n"
            "      <material name=\"red\">\n"
            "        <color rgba=\"1 0 0 1\"/>\n"
            "      </material>\n"
            "    </visual>\n"
            "  </link>\n"
            "</robot>\n"
        )
        count += 1
        o.close()

    # Generate the floor URDF
    o = open(os.path.join(ASSET_PATH, "floor.urdf"), "w")
    o.write(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"floor\">\n"
        "  <link name=\"floor\">\n"
        "    <collision>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 0 -5\"/>\n"
        "      <geometry>\n"
        "        <box size=\"" + str(map.size.width) + " " + str(map.size.height) + " 0\"/>\n"
        "      </geometry>\n"
        "    </collision>\n"
        "    <visual>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"" + str(map.size.width) + " " + str(map.size.height) + " 0\"/>\n"
        "      </geometry>\n"
        "      <material name=\"white\">\n"
        "        <color rgba=\"1 1 1 1\"/>\n"
        "      </material>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )

    # Generate bounds
    #region Bounds
    o = open(os.path.join(ASSET_PATH, "bound1.urdf"), "w")
    o.write(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"bound2\">\n"
        "  <link name=\"bound2\">\n"
        "    <collision>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 " + str(map.size.height/2) + " 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"" + str(map.size.width) + " 0 10\"/>\n"
        "      </geometry>\n"
        "    </collision>\n"
        "    <visual>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 " + str(map.size.height/2) + " 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"" + str(map.size.width) + " 0 10\"/>\n"
        "      </geometry>\n"
        "      <material name=\"red\">\n"
        "        <color rgba=\"1 0 0 1\"/>\n"
        "      </material>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )

    o = open(os.path.join(ASSET_PATH, "bound2.urdf"), "w")
    o.write(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"bound2\">\n"
        "  <link name=\"bound2\">\n"
        "    <collision>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 -" + str(map.size.height/2) + " 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"" + str(map.size.width) + " 0 10\"/>\n"
        "      </geometry>\n"
        "    </collision>\n"
        "    <visual>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 -" + str(map.size.height/2) + " 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"" + str(map.size.width) + " 0 10\"/>\n"
        "      </geometry>\n"
        "      <material name=\"red\">\n"
        "        <color rgba=\"1 0 0 1\"/>\n"
        "      </material>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )

    o = open(os.path.join(ASSET_PATH, "bound3.urdf"), "w")
    o.write(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"bound3\">\n"
        "  <link name=\"bound3\">\n"
        "    <collision>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"" + str(map.size.width/2) + " 0 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"0 " + str(map.size.height) + " 10\"/>\n"
        "      </geometry>\n"
        "    </collision>\n"
        "    <visual>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"" + str(map.size.width/2) + " 0 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"0 " + str(map.size.height) + " 10\"/>\n"
        "      </geometry>\n"
        "      <material name=\"red\">\n"
        "        <color rgba=\"1 0 0 1\"/>\n"
        "      </material>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )

    o = open(os.path.join(ASSET_PATH, "bound4.urdf"), "w")
    o.write(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"bound4\">\n"
        "  <link name=\"bound4\">\n"
        "    <collision>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"-" + str(map.size.width/2) + " 0 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"0 " + str(map.size.height) + " 10\"/>\n"
        "      </geometry>\n"
        "    </collision>\n"
        "    <visual>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"-" + str(map.size.width/2) + " 0 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"0 " + str(map.size.height) + " 10\"/>\n"
        "      </geometry>\n"
        "      <material name=\"red\">\n"
        "        <color rgba=\"1 0 0 1\"/>\n"
        "      </material>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )
    #endregion